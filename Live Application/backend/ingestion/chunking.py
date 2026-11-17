"""
Chunking: split each page into overlapping chunks and prepare what is stored in Pinecone.

    make_splitter()        RecursiveCharacterTextSplitter (CHUNK_SIZE / CHUNK_OVERLAP from config)
    leave_label / bank_label   human-readable label put in front of every chunk's text
    make_chunks(...)       chunks of one document; chunk hash (SHA-256 of the text) = vector id
    chunk_metadata(...)    everything retrieval needs, stored with the vector
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.utils import sha256
from guardrails import find_injection

SCHEMA_VERSION = 2       # bump when the metadata layout changes -> documents are re-processed


def make_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def leave_label(year, doc_id, info, page_no):
    return f"Leave Policy {year} ({doc_id}), page {page_no}"


def bank_label(year, doc_id, info, page_no):
    who = f" – {info['customer']}" if info.get("customer") else ""
    return f"{doc_id} {info['doc_type']}{who} [{info['classification']}], page {page_no}"


def make_chunks(path, year, doc_id, info, pages, splitter, label_fn, id_prefix, screen=False):
    """Chunks of one document. With screen=True (bank guardrail) chunks containing prompt-injection
    text are quarantined. Returns (chunks, quarantined [(page, matched text)])."""
    chunks, ids, quarantined = [], set(), []
    for page_no, text in pages:
        for i, piece in enumerate(splitter.split_text(text)):
            if screen:
                hit = find_injection(piece)
                if hit:
                    quarantined.append((page_no, hit))
                    continue
            label = label_fn(year, doc_id, info, page_no)
            full_text = f"{label}: {piece}"            # label inside the text so both searches see it
            chunk_id = id_prefix + sha256(full_text)[:40]   # chunk hash = vector id
            if chunk_id in ids:                        # identical chunk twice in one document
                continue
            ids.add(chunk_id)
            chunks.append({"id": chunk_id, "year": year, "page": page_no, "doc_id": doc_id,
                           "source": path.name, "label": label, "text": full_text,
                           "chunk_index": i,
                           "page_text": text if i == 0 else None})   # full page on the first chunk
    return chunks, quarantined


def chunk_metadata(c, doc_hash, doc_info, extra=None):
    """Everything retrieval needs, stored with the vector in Pinecone (no None values allowed)."""
    meta = {"text": c["text"], "label": c["label"], "year": c["year"], "page": c["page"],
            "doc_id": c["doc_id"], "source": c["source"], "chunk_index": c["chunk_index"],
            "doc_hash": doc_hash, "schema_version": SCHEMA_VERSION, **doc_info, **(extra or {})}
    if c["page_text"]:
        meta["page_text"] = c["page_text"]
    return meta
