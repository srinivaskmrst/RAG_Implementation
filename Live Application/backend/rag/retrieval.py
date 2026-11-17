"""
retrieve(): the search used by every pipeline (base, Corrective, Agentic).

    Traditional section -> traditional_rag.search   (vector)
    Hybrid section      -> hybrid_rag.search        (BM25 + vector, RRF)
    + Reranking         -> reranking.rerank          (when switched on)
    + bank guardrails   -> run.guard.screen          (access re-check, injection scan, PII re-mask)

Top-K values come from the UI: keyword_top_k (hybrid), vector_top_k, final_top_k.
"""

from rag import traditional_rag, hybrid_rag, reranking

SEARCHES = {"traditional": traditional_rag, "hybrid": hybrid_rag}


def retrieve(run, query, year=None):
    method = SEARCHES[run.mode]
    final_k = run.option("final_top_k")
    candidates, detail = method.search(run, query, year)

    if run.option("rerank"):
        ranked = reranking.rerank(run, query, year, candidates, final_k, method.SCORE)
        run.step(f"{method.NAME} → Rerank",
                 f"{detail}; re-scored by cross-encoder, kept top {len(ranked)}", kind="search")
        score_name = "rerank"
    else:
        ranked = candidates[:final_k]
        run.step(method.NAME, f"{detail}; kept top {len(ranked)}", kind="search")
        score_name = method.SCORE

    by_id = run.corpus.chunk_by_id
    chunks = [{**by_id[cid], "score": round(score, 4), "score_name": score_name}
              for cid, score in ranked if cid in by_id]
    if run.guard:          # retrieval guardrail: access re-check, injection scan, PII re-mask
        chunks = run.guard.screen(chunks)
    return chunks
