"""
Hybrid RAG retrieval: keyword search (BM25) + vector search (Pinecone), fused with
Reciprocal Rank Fusion (RRF).

    BM25 finds exact words (names, numbers, terms); vector search finds the meaning.
    RRF score of a chunk = sum over both result lists of 1 / (60 + rank).
The BM25 index is built from the chunk texts loaded from Pinecone (core/knowledge_base.py).
"""

from core.knowledge_base import kb
from core.utils import tokenize, matches_filter
from rag.traditional_rag import vector_search

NAME = "Hybrid search (BM25 + vector, RRF)"
SCORE = "rrf"
RRF_K = 60


def keyword_search(query, k=20, year=None, domain="leave", where=None):
    """[(chunk_id, BM25 score), ...] best first. `where` = bank access filter."""
    corpus = kb.corpus(domain)
    chunks, bm25 = corpus.keyword
    if not bm25:
        return []
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    ranked = [(c, s) for c, s in ranked if c["id"] in corpus.chunk_by_id
              and (not year or c["year"] == year) and matches_filter(c, where)]
    return [(c["id"], float(s)) for c, s in ranked[:k]]


def rrf_fuse(*result_lists):
    """Reciprocal Rank Fusion of several ranked lists -> [(chunk_id, rrf score), ...] best first."""
    rrf = {}
    for results in result_lists:
        for rank, (chunk_id, _score) in enumerate(results, start=1):
            rrf[chunk_id] = rrf.get(chunk_id, 0) + 1 / (RRF_K + rank)
    return sorted(rrf.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(query, year=None, keyword_k=20, vector_k=20, domain="leave", where=None):
    """Top keyword_k BM25 chunks + top vector_k Pinecone chunks, fused with RRF.
    Returns every unique chunk (up to keyword_k + vector_k), best first."""
    return rrf_fuse(keyword_search(query, k=keyword_k, year=year, domain=domain, where=where),
                    vector_search(query, k=vector_k, year=year, domain=domain, where=where))


def search(run, query, year=None):
    """The Hybrid RAG section's search. Returns (candidates, step detail)."""
    keyword_k, vector_k = run.option("keyword_top_k"), run.option("vector_top_k")
    candidates = hybrid_search(query, year=year, keyword_k=keyword_k, vector_k=vector_k,
                               domain=run.domain, where=run.where)
    year_note = f", year={year}" if year else ""
    return candidates, (f"“{query}”{year_note}: keyword top {keyword_k} + vector top {vector_k} "
                        f"→ RRF fusion → {len(candidates)} unique chunks")
