"""
RAG pipelines served to the UI.

Two base pipelines (one section each in the UI):
    traditional : question -> vector search (Pinecone) -> LLM answer
    hybrid      : question -> BM25 + Pinecone fused with RRF -> LLM answer

Optional techniques the user can switch on (applied to BOTH sections):
    rerank      : re-order the retrieved candidates with a cross-encoder, keep the best final_top_k
    corrective  : grade every retrieved chunk (relevant / partial / irrelevant); if nothing is
                  relevant, rewrite the question and search again (Corrective RAG)
    agentic     : AgenticRAG.py - a Google Gemini agent that decides itself which tools to call:
                  the company leave policy (this section's retriever, + rerank / grading if
                  selected) and live Wikipedia data through its public API (Agentic RAG).
"""

import time

from rag_core import kb, years_in, parse_json, LLM_PROVIDER, GEMINI_MODEL, model_label, default_model
from AgenticRAG import run_agent

# defaults only - the UI sends the top-K values with every question
DEFAULT_KEYWORD_TOP_K = 10
DEFAULT_VECTOR_TOP_K = 10
DEFAULT_FINAL_TOP_K = 5
MAX_REWRITES = 2

GRADE_ORDER = {"relevant": 2, "partial": 1, "irrelevant": 0}


class Run:
    """Collects the trace (steps, sources, LLM calls) of one pipeline run."""

    def __init__(self, mode, options):
        self.mode = mode
        self.options = options
        # provider + model chosen in the UI (Ollama or Google Gemini); defaults come from .env
        self.provider = options.get("provider") or LLM_PROVIDER
        self.model = options.get("model") or default_model(self.provider)
        self.label = model_label(self.model, self.provider)
        self.steps = []
        self.reranks = []          # before / after reranking tables, one per search
        self.llm_calls = 0
        self.started = time.perf_counter()

    def step(self, title, detail="", kind="info"):
        self.steps.append({"title": title, "detail": detail,
                           "kind": kind, "t": round(time.perf_counter() - self.started, 2)})

    def llm(self, prompt, as_json=False):
        self.llm_calls += 1
        return kb.ask_llm(prompt, as_json=as_json, model=self.model, provider=self.provider)


# ---------------------------------------------------------------------------- retrieval
def record_rerank(run, query, year, candidates, reranked, final_k, retrieval_score_name):
    """Save the ranking before reranking (search order) and after (cross-encoder order)
    so the UI can show how every candidate moved and which ones were kept."""
    before = {cid: (pos, score) for pos, (cid, score) in enumerate(candidates, start=1)}
    rows = []
    for after_pos, (cid, rerank_score) in enumerate(reranked, start=1):
        c = kb.chunk_by_id.get(cid)
        if not c:
            continue
        before_pos, retrieval_score = before[cid]
        rows.append({
            "label": f"Leave Policy {c['year']} · p.{c['page']}",
            "snippet": c["text"].split(": ", 1)[-1][:160],
            "before": before_pos, "after": after_pos,
            "move": before_pos - after_pos,                  # > 0 moved up, < 0 moved down
            "retrieval_score": round(retrieval_score, 4),
            "rerank_score": round(rerank_score, 4),
            "kept": after_pos <= final_k,
        })
    run.reranks.append({"query": query, "year": year, "retrieval_score_name": retrieval_score_name,
                        "candidates": len(rows), "kept": min(final_k, len(rows)), "rows": rows})


def retrieve(run, query, year=None):
    """Base search for the section (vector or hybrid), plus reranking when selected.
    The top-K values come from the UI:
        keyword_top_k : chunks taken from keyword search (BM25)        - hybrid only
        vector_top_k  : chunks taken from vector search (Pinecone)     - both sections
        final_top_k   : chunks kept after RRF fusion / reranking and sent to the LLM"""
    keyword_k = run.options.get("keyword_top_k", DEFAULT_KEYWORD_TOP_K)
    vector_k = run.options.get("vector_top_k", DEFAULT_VECTOR_TOP_K)
    final_k = run.options.get("final_top_k", DEFAULT_FINAL_TOP_K)
    year_note = f", year={year}" if year else ""

    if run.mode == "traditional":
        candidates = kb.vector_search(query, k=vector_k, year=year)
        search_name = "Vector search (Pinecone)"
        detail = f"“{query}”{year_note}: vector top {vector_k} → {len(candidates)} chunks"
        score_name = "cosine"
    else:
        candidates = kb.hybrid_search(query, year=year, keyword_k=keyword_k, vector_k=vector_k)
        search_name = "Hybrid search (BM25 + vector, RRF)"
        detail = (f"“{query}”{year_note}: keyword top {keyword_k} + vector top {vector_k} "
                  f"→ RRF fusion → {len(candidates)} unique chunks")
        score_name = "rrf"

    if run.options.get("rerank"):
        # score ALL candidates with the cross-encoder so every chunk gets a position after
        # reranking (for the before / after table in the UI), then keep the final top K
        full_order = kb.rerank(query, [cid for cid, _ in candidates], top_k=len(candidates))
        ranked = full_order[:final_k]
        record_rerank(run, query, year, candidates, full_order, final_k, score_name)
        run.step(f"{search_name} → Rerank",
                 f"{detail}; re-scored by cross-encoder, kept top {len(ranked)}", kind="search")
        score_name = "rerank"
    else:
        ranked = candidates[:final_k]
        run.step(search_name, f"{detail}; kept top {len(ranked)}", kind="search")

    return [{**kb.chunk_by_id[cid], "score": round(score, 4), "score_name": score_name}
            for cid, score in ranked if cid in kb.chunk_by_id]


# ---------------------------------------------------------------------------- corrective helpers
def grade_chunks(run, question, chunks):
    """Grade all chunks in one LLM call; adds 'grade' and 'evidence'."""
    if not chunks:
        return []
    listing = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(chunks, start=1))
    prompt = f"""You grade search results for a question about company leave policies.
Each policy applies to one calendar year only. A chunk from the wrong year is NOT relevant
when the question names a year. If the question names no year, the latest year is preferred.

For every chunk return:
  "id": the chunk number,
  "grade": "relevant"   - it directly contains the answer, for the right year
           "partial"    - related topic or useful background, but does not answer it
           "irrelevant" - does not help,
  "evidence": the exact sentence(s) from the chunk that help answer the question
              ("" when irrelevant).

Return JSON only, in this form:
{{"grades": [{{"id": 1, "grade": "relevant", "evidence": "..."}}, {{"id": 2, "grade": "irrelevant", "evidence": ""}}]}}

Question: {question}

Chunks:
{listing}"""
    grades = parse_json(run.llm(prompt, as_json=True), default=[])
    if isinstance(grades, dict):          # {"grades": [...]} (or any key holding the list)
        grades = grades.get("grades") or next((v for v in grades.values() if isinstance(v, list)), [])
    if not isinstance(grades, list):
        grades = []
    by_number = {str(g.get("id")): g for g in grades if isinstance(g, dict)}

    graded = []
    for n, c in enumerate(chunks, start=1):
        g = by_number.get(str(n), {})
        grade = str(g.get("grade", "irrelevant")).lower()
        graded.append({**c, "grade": grade if grade in GRADE_ORDER else "irrelevant",
                       "evidence": (g.get("evidence") or "").strip()})
    counts = {k: sum(c["grade"] == k for c in graded) for k in GRADE_ORDER}
    run.step("Grade chunks", f"{counts['relevant']} relevant · {counts['partial']} partial · "
                             f"{counts['irrelevant']} irrelevant", kind="grade")
    return graded


def decide(graded):
    if any(c["grade"] == "relevant" for c in graded):
        return "CORRECT"
    if any(c["grade"] == "partial" for c in graded):
        return "AMBIGUOUS"
    return "INCORRECT"


def rewrite_question(run, question, previous):
    prompt = f"""The search below did not find a good answer in the company leave policy
documents (one PDF per year, 2015-2026, with leave types such as casual, sick, earned,
maternity, paternity, caregiver, wellness, sabbatical and so on).

Rewrite the question as a better search query: use the formal policy terms for the leave
type, keep any year mentioned, and drop filler words. Give a different query from the
earlier attempts. Return only the query text.

Original question: {question}
Earlier queries: {previous}"""
    return run.llm(prompt).strip().strip('"') or question


def merge(kept, new):
    by_id = {c["id"]: c for c in kept}
    for c in new:
        if c["id"] not in by_id or GRADE_ORDER[c["grade"]] > GRADE_ORDER[by_id[c["id"]]["grade"]]:
            by_id[c["id"]] = c
    return list(by_id.values())


def corrective_retrieve(run, question):
    """Retrieve -> grade -> (rewrite -> retrieve -> grade)* until something is relevant."""
    queries, pool = [question], []
    for attempt in range(MAX_REWRITES + 1):
        pool = merge(pool, grade_chunks(run, question, retrieve(run, queries[-1])))
        verdict = decide(pool)
        run.step(f"Verdict: {verdict}", kind="ok" if verdict == "CORRECT" else "warn")
        if verdict == "CORRECT" or attempt == MAX_REWRITES:
            break
        queries.append(rewrite_question(run, question, queries))
        run.step("Rewrite question", f"→ “{queries[-1]}”", kind="rewrite")

    useful = [c for c in pool if c["grade"] != "irrelevant"]
    useful.sort(key=lambda c: (-GRADE_ORDER[c["grade"]], -(c["year"] or 0)))
    return useful, decide(pool)


# ---------------------------------------------------------------------------- answer
def generate_answer(run, question, chunks):
    context = "\n\n".join(f"[Source {n}] {c['label']}:\n{c.get('evidence') or c['text'].split(': ', 1)[-1]}"
                          for n, c in enumerate(chunks, start=1))
    prompt = f"""You are an HR assistant for Acme Technologies India Pvt. Ltd.
Answer the question using only the sources below. These are leave policies for different
years; use the policy of the year the question asks about. If no year is asked, use the latest
year in the sources and say which year. Be concise, give exact numbers and conditions.
Cite the sources you used, e.g. [Source 2]. If the sources do not contain the answer, say so.

{context}

Question: {question}"""
    answer = run.llm(prompt)
    run.step("Generate answer", f"{run.label} with {len(chunks)} sources", kind="llm")
    return answer


# ---------------------------------------------------------------------------- entry point
def run_pipeline(mode, question, options):
    if options.get("agentic"):
        # Agentic RAG always runs on Google Gemini (function calling): use the model chosen in
        # the Gemini dropdown, even when the Ollama card is active for the other options
        options = {**options, "provider": "gemini",
                   "model": options.get("gemini_model") or GEMINI_MODEL}
    run = Run(mode, options)
    techniques = [name for name, on in (("Rerank", options.get("rerank")),
                                        ("Corrective", options.get("corrective")),
                                        ("Agentic", options.get("agentic"))) if on]
    verdict = None
    web_sources = []

    if options.get("agentic"):
        answer, sources, web_sources = run_agent(run, question, retrieve, grade_chunks, generate_answer)
    elif options.get("corrective"):
        sources, verdict = corrective_retrieve(run, question)
        if not sources:
            run.step("No useful chunks", "answer not found in the policies", kind="warn")
            answer = "Answer not found. The Leave Policy documents (2015-2026) do not contain the answer."
        else:
            answer = generate_answer(run, question, sources)
    else:
        sources = retrieve(run, question)
        answer = generate_answer(run, question, sources)

    return {
        "mode": mode,
        "model": run.label,
        "provider": run.provider,
        "techniques": techniques,
        "answer": answer,
        "verdict": verdict,
        "steps": run.steps,
        "reranking": run.reranks,
        "sources": [{"label": c["label"], "year": c["year"], "page": c["page"],
                     "score": c.get("score"), "score_name": c.get("score_name"),
                     "grade": c.get("grade"),
                     "text": c.get("evidence") or c["text"].split(": ", 1)[-1]}
                    for c in sources]
                   + [{"kind": "web", "title": w["title"], "url": w["url"], "text": w["text"]}
                      for w in web_sources],
        "llm_calls": run.llm_calls,
        "seconds": round(time.perf_counter() - run.started, 2),
        "years_asked": sorted(years_in(question)),
        "top_k": {"keyword": options.get("keyword_top_k", DEFAULT_KEYWORD_TOP_K) if mode == "hybrid" else None,
                  "vector": options.get("vector_top_k", DEFAULT_VECTOR_TOP_K),
                  "final": options.get("final_top_k", DEFAULT_FINAL_TOP_K)},
    }
