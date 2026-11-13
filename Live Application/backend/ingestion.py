"""
Incremental ingestion: documents -> chunks -> embeddings -> Pinecone, with two hash checks.

1. Document hash = SHA-256 of the file's bytes. It is stored in the metadata of every chunk of
   that document ("doc_hash"). If Pinecone already holds vectors with this doc_hash, the document
   is unchanged and is skipped completely (no embedding, no upload).

2. Chunk hash = SHA-256 of the chunk text, used as the vector id. When a document is new or its
   hash changed, every chunk id is looked up in Pinecone:
     - chunk id already in Pinecone  -> skipped (only its metadata is pointed at the new doc_hash)
     - chunk id not in Pinecone      -> embedded and inserted
     - old chunk of this document that no longer exists in the new version -> deleted
   Vectors of files that were removed from the folder are deleted as well.

Search keeps working during ingestion: the new in-memory chunks / BM25 index are swapped in at
the end (see KnowledgeBase.ingest in rag_core.py).
"""

import hashlib
import json
import re
import time
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_core import DATA_FOLDER, NAMESPACE, CHUNK_SIZE, CHUNK_OVERLAP

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
BATCH = 100              # Pinecone upsert / fetch / delete batch size
MAX_IDS = 10_000         # Pinecone top_k limit when values and metadata are not returned


def sha256(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------- reading
def read_pages(path):
    """List of (page_number, text) for one file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [(n, page.extract_text() or "") for n, page in enumerate(reader.pages, start=1)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def describe(path, text, manifest):
    """Year and document id: from manifest.json, else detected from the file name / text."""
    entry = manifest.get(path.name)
    if entry:
        return int(entry["year"]), entry["doc_id"]
    in_name = re.search(r"(?:19|20)\d{2}", path.stem)                     # Leave_Policy_2019.pdf
    in_text = re.search(r"Policy year\s*((?:19|20)\d{2})", text)          # "Policy year 2019"
    year = int(in_name.group(0)) if in_name else int(in_text.group(1)) if in_text else 0
    doc_id = re.search(r"HR-LV-\d{4}", text)
    return year, doc_id.group(0) if doc_id else path.stem


def make_chunks(path, year, doc_id, pages, splitter):
    chunks, ids = [], set()
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            label = f"Leave Policy {year} ({doc_id}), page {page_no}"
            full_text = f"{label}: {piece}"            # label inside the text so both searches see the year
            chunk_id = "c-" + sha256(full_text)[:40]   # chunk hash = vector id
            if chunk_id in ids:                        # identical chunk twice in one document
                continue
            ids.add(chunk_id)
            chunks.append({"id": chunk_id, "year": year, "page": page_no, "doc_id": doc_id,
                           "source": path.name, "label": label, "text": full_text})
    return chunks


# ---------------------------------------------------------------------------- Pinecone helpers
def ids_matching(kb, probe, metadata_filter, top_k=MAX_IDS, include_metadata=False):
    """Ids of vectors matching a metadata filter (a query with any vector + filter)."""
    result = kb.index.query(vector=probe, top_k=top_k, namespace=NAMESPACE,
                            filter=metadata_filter, include_metadata=include_metadata)
    return result.matches


def existing_ids(kb, ids):
    found = set()
    for start in range(0, len(ids), BATCH):
        response = kb.index.fetch(ids=ids[start:start + BATCH], namespace=NAMESPACE)
        found.update(response.vectors.keys())
    return found


def upsert_chunks(kb, chunks, doc_hash):
    if not chunks:
        return
    vectors = kb.embed([c["text"] for c in chunks])
    records = [{"id": c["id"], "values": vectors[i].tolist(),
                "metadata": {"year": c["year"], "page": c["page"], "doc_id": c["doc_id"],
                             "source": c["source"], "doc_hash": doc_hash}}
               for i, c in enumerate(chunks)]
    for start in range(0, len(records), BATCH):
        kb.index.upsert(vectors=records[start:start + BATCH], namespace=NAMESPACE)


def delete_ids(kb, ids):
    ids = list(ids)
    for start in range(0, len(ids), BATCH):
        kb.index.delete(ids=ids[start:start + BATCH], namespace=NAMESPACE)


# ---------------------------------------------------------------------------- main
def run_ingestion(kb, log=print):
    started = time.perf_counter()
    manifest_path = DATA_FOLDER / "manifest.json"
    manifest_list = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest = {d["file"]: d for d in manifest_list}

    files = sorted(p for p in DATA_FOLDER.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    probe = kb.embed(["leave policy"])[0].tolist()   # any vector works for filter-only queries

    documents, all_pages, all_chunks, hash_to_file = [], [], [], {}

    for path in files:
        doc_hash = sha256(path.read_bytes())
        pages = read_pages(path)
        year, doc_id = describe(path, "\n".join(t for _, t in pages), manifest)
        row = {"file": path.name, "year": year, "doc_id": doc_id, "doc_hash": doc_hash[:12],
               "pages": len(pages), "chunks": 0, "added": 0, "existing": 0, "removed": 0,
               "status": "", "note": ""}
        documents.append(row)

        if doc_hash in hash_to_file:     # an identical copy of a file already processed in this run
            row.update(status="duplicate", note=f"Same content as {hash_to_file[doc_hash]} - skipped")
            continue
        hash_to_file[doc_hash] = path.name

        chunks = make_chunks(path, year, doc_id, pages, splitter)
        row["chunks"] = len(chunks)
        all_pages += [{"year": year, "doc_id": doc_id, "page": n, "text": t, "source": path.name}
                      for n, t in pages]
        all_chunks += chunks

        # ---- check 1: document hash
        if ids_matching(kb, probe, {"doc_hash": {"$eq": doc_hash}}, top_k=1):
            row.update(status="unchanged", existing=len(chunks),
                       note="Document hash already in vector DB - skipped")
            log(f"  {path.name}: unchanged")
            continue

        # ---- check 2: chunk hashes (document is new or changed)
        previous = {m.id for m in ids_matching(kb, probe, {"source": {"$eq": path.name}})}
        chunk_ids = [c["id"] for c in chunks]
        already = existing_ids(kb, chunk_ids)
        new_chunks = [c for c in chunks if c["id"] not in already]

        upsert_chunks(kb, new_chunks, doc_hash)
        for chunk_id in already:         # same chunk text: keep the vector, link it to the new doc hash
            kb.index.update(id=chunk_id, set_metadata={"doc_hash": doc_hash, "source": path.name},
                            namespace=NAMESPACE)
        stale = previous - set(chunk_ids)
        delete_ids(kb, stale)

        status = "changed" if previous else "new"
        row.update(status=status, added=len(new_chunks), existing=len(already), removed=len(stale),
                   note=(f"{len(new_chunks)} new chunks embedded, {len(already)} already in vector DB"
                         + (f", {len(stale)} outdated chunks removed" if stale else "")))
        log(f"  {path.name}: {status} (+{len(new_chunks)} / ={len(already)} / -{len(stale)})")

    # ---- files that were removed from the folder
    names = [p.name for p in files]
    orphans = {m.id for m in ids_matching(kb, probe, {"source": {"$nin": names}})} if names else set()
    delete_ids(kb, orphans)

    # Pinecone is eventually consistent: wait briefly until the count matches
    expected = len(all_chunks)
    vectors = kb.count_vectors()
    for _ in range(15):
        if vectors == expected:
            break
        time.sleep(2)
        vectors = kb.count_vectors()

    count = lambda status: sum(d["status"] == status for d in documents)
    summary = {
        "documents": len(documents),
        "new": count("new"), "changed": count("changed"),
        "unchanged": count("unchanged"), "duplicate": count("duplicate"),
        "pages": len(all_pages),
        "chunks": len(all_chunks),
        "chunks_added": sum(d["added"] for d in documents),
        "chunks_existing": sum(d["existing"] for d in documents),
        "chunks_removed": sum(d["removed"] for d in documents) + len(orphans),
        "orphans_removed": len(orphans),
        "vectors_in_db": vectors,
        "seconds": round(time.perf_counter() - started, 2),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "namespace": NAMESPACE,
    }
    state = {"manifest": manifest_list, "pages": all_pages, "chunks": all_chunks}
    return {"summary": summary, "documents": documents}, state
