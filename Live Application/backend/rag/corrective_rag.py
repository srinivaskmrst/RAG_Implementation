"""
Corrective RAG: check the retrieved chunks before answering.

    retrieve -> grade every chunk with the LLM (relevant / partial / irrelevant)
             -> verdict: CORRECT (something relevant) | AMBIGUOUS (only partial) | INCORRECT
             -> not CORRECT: rewrite the question into a better search query and search again
                (max MAX_REWRITES times)
             -> answer only from the useful chunks (grade relevant or partial)
"""

from core.config import MAX_REWRITES
from core.utils import parse_json
from rag.retrieval import retrieve

GRADE_ORDER = {"relevant": 2, "partial": 1, "irrelevant": 0}


def grade_chunks(run, question, chunks):
    """Grade all chunks in one LLM call; adds 'grade' and 'evidence'."""
    if not chunks:
        return []
    listing = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(chunks, start=1))
    prompt = f"""{run.prompts["grade"]}

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
    prompt = f"""{run.prompts["rewrite"]} Give a different query from the
earlier attempts. Return only the query text.

Original question: {question}
Earlier queries: {previous}"""
    return run.llm(prompt).strip().strip('"') or question


def merge(kept, new):
    """Keep every chunk once, with its best grade."""
    by_id = {c["id"]: c for c in kept}
    for c in new:
        if c["id"] not in by_id or GRADE_ORDER[c["grade"]] > GRADE_ORDER[by_id[c["id"]]["grade"]]:
            by_id[c["id"]] = c
    return list(by_id.values())


def corrective_retrieve(run, question):
    """Retrieve -> grade -> (rewrite -> retrieve -> grade)* until something is relevant.
    Returns (useful chunks, verdict)."""
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
