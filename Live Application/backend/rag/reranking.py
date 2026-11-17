"""
Reranking: a cross-encoder (BAAI/bge-reranker-base) reads the question and each retrieved chunk
TOGETHER and gives a relevance score - more accurate than the first search, but slower, so it
only re-orders the candidates the search found. For the leave policy the asked year comes first.

    Reranker      the model (loaded once at startup by core/knowledge_base.py)
    rerank(...)   re-order the candidates, keep the final top K, record the before / after table
"""

import threading

from core.config import RERANK_MODEL
from core.knowledge_base import kb
from core.utils import years_in


class Reranker:
    def __init__(self, model_name=RERANK_MODEL):
        import torch
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid())
        self._lock = threading.Lock()            # torch models are not thread-safe

    def score(self, query, texts):
        with self._lock:
            return self.model.predict([(query, t) for t in texts])


def rank(query, chunk_ids, domain="leave", top_k=None):
    """[(chunk_id, rerank score), ...] - asked year first (leave policy), then highest score."""
    by_id = kb.corpus(domain).chunk_by_id
    chunk_ids = [i for i in chunk_ids if i in by_id]   # an ingestion may have just swapped data
    if not chunk_ids:
        return []
    scores = kb.reranker.score(query, [by_id[i]["text"] for i in chunk_ids])
    asked = years_in(query) if domain == "leave" else set()

    def sort_key(pair):
        chunk_id, score = pair
        wrong_year = bool(asked) and by_id[chunk_id]["year"] not in asked
        return (wrong_year, -score)

    ranked = sorted(zip(chunk_ids, scores), key=sort_key)
    return [(i, float(s)) for i, s in ranked[:top_k]]


def rerank(run, query, year, candidates, final_k, retrieval_score_name):
    """Score ALL candidates with the cross-encoder (so every chunk gets a position for the
    before / after table in the UI) and keep the final top K."""
    from rag.run import short_label
    full_order = rank(query, [cid for cid, _ in candidates], domain=run.domain)

    before = {cid: (pos, score) for pos, (cid, score) in enumerate(candidates, start=1)}
    rows = []
    for after_pos, (cid, rerank_score) in enumerate(full_order, start=1):
        c = run.corpus.chunk_by_id.get(cid)
        if not c:
            continue
        before_pos, retrieval_score = before[cid]
        rows.append({
            "label": short_label(c),
            "snippet": c["text"].split(": ", 1)[-1][:160],
            "before": before_pos, "after": after_pos,
            "move": before_pos - after_pos,                  # > 0 moved up, < 0 moved down
            "retrieval_score": round(retrieval_score, 4),
            "rerank_score": round(rerank_score, 4),
            "kept": after_pos <= final_k,
        })
    run.reranks.append({"query": query, "year": year, "retrieval_score_name": retrieval_score_name,
                        "candidates": len(rows), "kept": min(final_k, len(rows)), "rows": rows})
    return full_order[:final_k]
