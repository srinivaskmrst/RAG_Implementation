"""
Traditional RAG retrieval: vector (meaning) search in Pinecone.

    question -> embedding (core/embedding.py) -> Pinecone query (top K, optional year / access
    filter) -> chunks ranked by cosine similarity
"""

from core.knowledge_base import kb, chunk_from_metadata
from core.utils import matches_filter

NAME = "Vector search (Pinecone)"
SCORE = "cosine"


def vector_search(query, k=20, year=None, domain="leave", where=None):
    """[(chunk_id, cosine score), ...] best first. `where` = bank access filter."""
    corpus = kb.corpus(domain)
    vector = kb.embedder.embed(query).tolist()
    flt = {**({"year": {"$eq": year}} if year else {}), **(where or {})}
    found = []
    for m in kb.store.query(corpus.namespace, vector, k, flt, include_metadata=True):
        if m.id not in corpus.chunk_by_id and (m.metadata or {}).get("text"):
            # chunk added to Pinecone after the last reload: take it straight from its metadata
            corpus.chunk_by_id[m.id] = chunk_from_metadata(m.id, m.metadata)
        if m.id in corpus.chunk_by_id and matches_filter(corpus.chunk_by_id[m.id], where):
            found.append((m.id, float(m.score)))
    return found


def search(run, query, year=None):
    """The Traditional RAG section's search. Returns (candidates, step detail)."""
    vector_k = run.option("vector_top_k")
    candidates = vector_search(query, k=vector_k, year=year, domain=run.domain, where=run.where)
    year_note = f", year={year}" if year else ""
    return candidates, f"“{query}”{year_note}: vector top {vector_k} → {len(candidates)} chunks"
