"""
Incremental ingestion of one dataset: folder -> chunks -> embeddings -> Pinecone namespace.

Two hash checks keep ingestion incremental:

1. Document hash = SHA-256 of the file's bytes, stored as "doc_hash" on every chunk of that
   document. If Pinecone already holds vectors with this doc_hash (and the current metadata
   schema), the document is unchanged and is skipped completely.

2. Chunk hash = SHA-256 of the chunk text, used as the vector id. When a document is new or its
   hash changed, every chunk id is looked up in Pinecone:
     - chunk id already in Pinecone  -> not embedded again; only its metadata is refreshed
     - chunk id not in Pinecone      -> embedded and inserted
     - old chunk of this document that no longer exists in the new version -> deleted
   Vectors of files that were removed from the folder are deleted as well.

Bank ingestion guardrails (guardrails package, Data Privacy SOP DOC-026):
   - PII masked BEFORE chunking and embedding - raw PII never reaches Pinecone, the LLM or the UI
   - chunks containing prompt-injection text are QUARANTINED (not stored)
   - classification label kept (missing -> CONFIDENTIAL) + customer / category for access filters
   - files over MAX_FILE_MB are rejected
"""

import time

from core.config import DOMAINS
from core.utils import sha256
import guardrails
from ingestion.document_loader import list_files, read_pages, load_manifest, describe_leave, describe_bank
from ingestion.chunking import (SCHEMA_VERSION, make_splitter, leave_label, bank_label, make_chunks,
                                chunk_metadata)

MAX_FILE_MB = 20         # bank guardrail: larger files are rejected


def embed_and_upsert(kb, namespace, chunks, doc_hash, doc_info, extra):
    """Embed new chunks (core/embedding.py) and insert them into Pinecone (core/vector_store.py)."""
    if not chunks:
        return
    vectors = kb.embedder.embed([c["text"] for c in chunks])
    kb.store.upsert(namespace, [{"id": c["id"], "values": vectors[i].tolist(),
                                 "metadata": chunk_metadata(c, doc_hash, doc_info, extra)}
                                for i, c in enumerate(chunks)])


def mask_pages(pages, totals):
    """Bank guardrail: mask PII on every page. Returns (masked pages, {pii type: count})."""
    masked_pages, pii = [], {}
    for page_no, text in pages:
        masked, counts = guardrails.mask_pii(text)
        masked_pages.append((page_no, masked))
        for k, v in counts.items():
            pii[k] = pii.get(k, 0) + v
            totals[k] = totals.get(k, 0) + v
    return masked_pages, pii


def run_ingestion(kb, domain="leave", log=print):
    started = time.perf_counter()
    cfg = DOMAINS[domain]
    folder, namespace, guarded = cfg["folder"], cfg["namespace"], cfg["guardrails"]
    store = kb.store

    files = list_files(folder)
    manifest = load_manifest(folder)
    splitter = make_splitter()
    probe = kb.embedder.embed(["document"])[0].tolist()   # any vector works for filter-only queries
    # the guard version is part of the "unchanged" check: new masking rules -> documents re-processed
    extra = {"guard_version": guardrails.GUARD_VERSION} if guarded else {}
    id_prefix = "b-" if guarded else "c-"

    documents, total_pages, total_chunks, hash_to_file = [], 0, 0, {}
    pii_total, quarantined_total, rejected = {}, 0, 0

    for path in files:
        raw = path.read_bytes()
        doc_hash = sha256(raw)
        row = {"file": path.name, "year": None, "doc_id": "", "doc_hash": doc_hash[:12],
               "pages": 0, "chunks": 0, "added": 0, "existing": 0, "removed": 0,
               "status": "", "note": ""}
        documents.append(row)

        if guarded and len(raw) > MAX_FILE_MB * 1_000_000:          # guardrail: file size
            row.update(status="rejected", note=f"Guardrail: file larger than {MAX_FILE_MB} MB - not ingested")
            rejected += 1
            continue

        # ---- read + describe (+ mask PII for the bank)
        pages = read_pages(path)
        text = "\n".join(t for _, t in pages)
        notes = []
        if guarded:
            year, doc_id, doc_info, labelled = describe_bank(path, text)
            pages, pii = mask_pages(pages, pii_total)              # guardrail: PII masking
            row.update(customer=doc_info.get("customer", ""), doc_type=doc_info["doc_type"],
                       classification=doc_info["classification"], pii_masked=sum(pii.values()),
                       pii_types=pii)
            if not labelled:
                notes.append("no classification label - stored as CONFIDENTIAL")
            label_fn = bank_label
        else:
            year, doc_id, doc_info = describe_leave(path, text, manifest)
            label_fn = leave_label
        row.update(year=year or None, doc_id=doc_id, pages=len(pages))

        if doc_hash in hash_to_file:     # an identical copy of a file already processed in this run
            row.update(status="duplicate", note=f"Same content as {hash_to_file[doc_hash]} - skipped")
            continue
        hash_to_file[doc_hash] = path.name

        # ---- chunk (+ injection scan for the bank)
        chunks, quarantined = make_chunks(path, year, doc_id, doc_info, pages, splitter, label_fn,
                                          id_prefix, screen=guarded)
        if quarantined:                                             # guardrail: injection scan
            quarantined_total += len(quarantined)
            row["quarantined"] = len(quarantined)
            notes.append(f"{len(quarantined)} chunk(s) quarantined (prompt injection: "
                         f"“{quarantined[0][1]}”, p.{quarantined[0][0]})")
        row["chunks"] = len(chunks)
        total_pages += len(pages)
        total_chunks += len(chunks)

        # ---- check 1: document hash (+ current metadata schema / guardrail version)
        if store.ids_matching(namespace, probe, {"doc_hash": {"$eq": doc_hash},
                                                 "schema_version": {"$eq": SCHEMA_VERSION},
                                                 **{k: {"$eq": v} for k, v in extra.items()}}, top_k=1):
            row.update(status="unchanged", existing=len(chunks),
                       note="; ".join(["Document hash already in vector DB - skipped"] + notes))
            log(f"  {path.name}: unchanged")
            continue

        # ---- check 2: chunk hashes (document is new, changed, or stored with an old schema)
        previous = store.ids_matching(namespace, probe, {"source": {"$eq": path.name}})
        chunk_ids = [c["id"] for c in chunks]
        already = store.existing_ids(namespace, chunk_ids)
        new_chunks = [c for c in chunks if c["id"] not in already]

        embed_and_upsert(kb, namespace, new_chunks, doc_hash, doc_info, extra)
        for c in chunks:                 # same chunk text: keep the vector, refresh its metadata
            if c["id"] in already:
                store.update_metadata(namespace, c["id"], chunk_metadata(c, doc_hash, doc_info, extra))
        stale = previous - set(chunk_ids)
        store.delete_ids(namespace, stale)

        status = "changed" if previous else "new"
        row.update(status=status, added=len(new_chunks), existing=len(already), removed=len(stale),
                   note="; ".join([f"{len(new_chunks)} new chunks embedded, {len(already)} already in vector DB"
                                   + (" (metadata refreshed)" if already else "")
                                   + (f", {len(stale)} outdated chunks removed" if stale else "")] + notes))
        log(f"  {path.name}: {status} (+{len(new_chunks)} / ={len(already)} / -{len(stale)})")

    # ---- files that were removed from the folder (or rejected by a guardrail)
    names = [d["file"] for d in documents if d["status"] != "rejected"]
    orphans = store.ids_matching(namespace, probe, {"source": {"$nin": names}}) if names else set()
    store.delete_ids(namespace, orphans)

    vectors = store.wait_for_count(namespace, total_chunks)

    count = lambda status: sum(d["status"] == status for d in documents)
    summary = {
        "domain": domain, "name": cfg["name"], "folder": folder.name,
        "documents": len(documents),
        "new": count("new"), "changed": count("changed"),
        "unchanged": count("unchanged"), "duplicate": count("duplicate"),
        "pages": total_pages,
        "chunks": total_chunks,
        "chunks_added": sum(d["added"] for d in documents),
        "chunks_existing": sum(d["existing"] for d in documents),
        "chunks_removed": sum(d["removed"] for d in documents) + len(orphans),
        "orphans_removed": len(orphans),
        "vectors_in_db": vectors,
        "seconds": round(time.perf_counter() - started, 2),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "namespace": namespace,
        "index": kb.index_name,
        "guardrails": guarded,
    }
    if guarded:
        summary.update(pii_masked=sum(pii_total.values()), pii_types=pii_total,
                       quarantined=quarantined_total, rejected=rejected,
                       customers=len({d.get("customer") for d in documents if d.get("customer")}))
    return {"summary": summary, "documents": documents}
