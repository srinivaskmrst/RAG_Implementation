"""
RAG pipelines served to the UI.

Two base pipelines (one section each in the UI):
    traditional : question -> vector search (Pinecone) -> LLM answer
    hybrid      : question -> BM25 + Pinecone fused with RRF -> LLM answer

Optional techniques the user can switch on (applied to BOTH sections):
    rerank      : re-order the retrieved candidates with a cross-encoder, keep the best final_top_k
    corrective  : grade every retrieved chunk (relevant / partial / irrelevant); if nothing is
                  relevant, rewrite the question and search again (Corrective RAG)
    agentic     : an LLM agent with tools decides what to search (and for which year),
                  can search several times / read full pages, then answers (Agentic RAG).
                  Its search tool uses the section's retriever (+ rerank / grading if selected).
"""

import time

from google.genai import types

import json

from rag_core import kb, years_in, parse_json, LLM_PROVIDER, model_label, default_model

# defaults only - the UI sends the top-K values with every question
DEFAULT_KEYWORD_TOP_K = 10
DEFAULT_VECTOR_TOP_K = 10
DEFAULT_FINAL_TOP_K = 5
MAX_REWRITES = 2
AGENT_MAX_STEPS = 8

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


# ---------------------------------------------------------------------------- agentic
AGENT_PROMPT = """You are an HR assistant for Acme Technologies India Pvt. Ltd. You answer employee
questions about the company leave policy using ONLY the policy documents, reached through your tools.
There is one leave policy per calendar year (2015-2026); entitlements change from year to year.

How to work:
1. Plan: work out which leave type(s) and which year(s) the question is about. If no year is
   given, assume the current policy (2026) and say so.
2. Retrieve: call search_leave_policy with focused queries in formal policy wording, using the
   year filter whenever a year is known. For comparisons across years, use
   list_policy_documents for the headline CL/SL/EL numbers and issue the per-year searches
   you need IN THE SAME TURN (several tool calls at once), not one year per turn.
   You have a limited number of tool rounds, so be efficient.
3. Reflect: if the passages do not answer the question, rephrase, try another year, or read the
   full page with read_policy_page. Do not stop at a weak result.
4. Answer: be concise, give exact numbers and conditions, and cite every fact as
   [HR-LV-YYYY, p.N]. If the documents do not contain the answer, say so - never guess."""


def make_tools(run, question, seen):
    """The agent's three tools. 'seen' collects every passage the agent looked at (sources panel)."""

    def list_policy_documents() -> dict:
        """List every leave policy document: year, document id, effective dates and the
        headline Casual (CL), Sick (SL) and Earned (EL) leave entitlements."""
        run.step("Tool: list_policy_documents", kind="tool")
        return {"documents": [{k: d[k] for k in ("year", "doc_id", "effective_from", "effective_to",
                                                  "cl", "sl", "el")} for d in kb.manifest]}

    def search_leave_policy(query: str, year: int = 0) -> dict:
        """Search the company leave policy documents (one PDF per year, 2015-2026) and return
        the most relevant passages with their year and page.

        Args:
            query: What to look for, in policy wording, e.g. "earned leave carry forward limit".
            year: Policy year to restrict the search to (e.g. 2019). Use 0 to search all years.
        """
        try:
            year = int(year or 0)
        except (TypeError, ValueError):     # small models sometimes send "all" or "2019 policy"
            found = years_in(str(year))
            year = found.pop() if found else 0
        run.step("Tool: search_leave_policy", f"query=“{query}”, year={year or 'all'}", kind="tool")
        if year and year not in kb.years:
            return {"error": f"No policy for {year}. Available years: {kb.years}"}
        chunks = retrieve(run, str(query), year=year or None)
        if run.options.get("corrective"):      # self-corrective agent: hand back only graded-useful chunks
            chunks = [c for c in grade_chunks(run, question, chunks) if c["grade"] != "irrelevant"]
            if not chunks:
                return {"results": [], "note": "No relevant passages. Rephrase the query or try another year."}
        for c in chunks:
            seen.setdefault(c["id"], c)
        return {"results": [{"source": c["label"], "year": c["year"], "page": c["page"],
                             "grade": c.get("grade"), "text": c["text"].split(": ", 1)[-1]}
                            for c in chunks]}

    def read_policy_page(year: int, page: int) -> dict:
        """Return the full text of one page of a leave policy PDF, for more context.

        Args:
            year: Policy year, e.g. 2022.
            page: Page number, starting at 1.
        """
        year, page = int(year), int(page)
        run.step("Tool: read_policy_page", f"year={year}, page={page}", kind="tool")
        for p in kb.pages:
            if p["year"] == year and p["page"] == page:
                return {"source": f"Leave Policy {year} ({p['doc_id']}), page {page}", "text": p["text"]}
        return {"error": f"Page {page} of {year} not found."}

    return {f.__name__: f for f in (list_policy_documents, search_leave_policy, read_policy_page)}


def call_tool(tools, name, args):
    if name not in tools:
        return {"error": f"Unknown tool '{name}'. Use one of: {', '.join(tools)}"}
    try:
        return tools[name](**(args if isinstance(args, dict) else {}))
    except Exception as e:  # report tool errors back to the model so it can recover
        return {"error": f"{type(e).__name__}: {e}"}


def gemini_agent(run, question, tools):
    """Gemini with native function calling."""
    config = types.GenerateContentConfig(
        system_instruction=AGENT_PROMPT, temperature=0, tools=list(tools.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    history = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    run.step("Agent planning", f"{run.label} with 3 tools, max {AGENT_MAX_STEPS} rounds", kind="llm")

    for _ in range(AGENT_MAX_STEPS):
        run.llm_calls += 1
        response = kb.generate(history, config, model=run.model)
        history.append(response.candidates[0].content)
        if not response.function_calls:
            return response.text
        parts = [types.Part.from_function_response(
                     name=fc.name, response=call_tool(tools, fc.name, dict(fc.args or {})))
                 for fc in response.function_calls]
        history.append(types.Content(role="user", parts=parts))
    return None


JSON_AGENT_PROMPT = """You are an HR assistant for Acme Technologies India Pvt. Ltd. You answer questions
about the company leave policy. There is one policy document per year (2015-2026) and the rules change
from year to year. You can only learn what the policy says by using tools.

Tools:
- search_leave_policy   args: {"query": "<search words>", "year": <policy year, or 0 for all years>}
- read_policy_page      args: {"year": <policy year>, "page": <page number>}
- list_policy_documents args: {}   (headline Casual, Sick and Earned leave days for every year)

Reply with ONE JSON object and nothing else.
To use a tool: {"thought": "<short reasoning>", "action": "<tool name>", "args": {...}}
To finish:     {"thought": "<short reasoning>", "action": "final_answer", "answer": "<your answer>"}

Rules: always search before answering. Use the year filter when the question names a year; if no
year is given, use 2026 and say so. Give exact numbers and conditions and cite facts as
[HR-LV-YYYY, p.N]. If the documents do not contain the answer, say so - never guess."""

JSON_AGENT_MAX_STEPS = 5   # local CPU models are slow, keep the loop short


def compact(result):
    """Shorten tool results so they fit in a small model's context window."""
    if "results" in result:
        return {**result, "results": [{**r, "text": r["text"][:600]} for r in result["results"][:4]]}
    if "text" in result:
        return {**result, "text": result["text"][:2000]}
    return result


def json_agent(run, question, tools):
    """Prompt-based agent for open-source models without native tool calling (Ollama):
    the model replies with a JSON action, we run the tool and send back the observation."""
    messages = [{"role": "system", "content": JSON_AGENT_PROMPT},
                {"role": "user", "content": f"Question: {question}"}]
    run.step("Agent planning", f"{run.label} with 3 tools (JSON actions), max {JSON_AGENT_MAX_STEPS} rounds",
             kind="llm")
    used_tool = False

    for _ in range(JSON_AGENT_MAX_STEPS):
        run.llm_calls += 1
        raw = kb.chat(messages, as_json=True, model=run.model)
        messages.append({"role": "assistant", "content": raw})
        reply = parse_json(raw, default={})
        if not isinstance(reply, dict):
            reply = {}
        action = str(reply.get("action") or "").strip()
        args = reply.get("args") if isinstance(reply.get("args"), dict) else {}
        if reply.get("thought"):
            run.step("Agent thought", str(reply["thought"])[:200], kind="llm")

        if action == "final_answer" or (not action and reply.get("answer")):
            answer = reply.get("answer") or args.get("answer")
            if not used_tool:      # small models like to answer from memory - make it search first
                messages.append({"role": "user", "content": "You have not searched the documents yet. "
                                 "Use search_leave_policy first, then answer."})
                continue
            if answer:
                return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)

        if action in tools:
            used_tool = True
            observation = compact(call_tool(tools, action, args))
            messages.append({"role": "user", "content": "Observation: " + json.dumps(observation, ensure_ascii=False)
                             + "\nReply with the next JSON object (another tool, or final_answer)."})
        else:
            messages.append({"role": "user", "content": 'Invalid reply. Reply with ONE JSON object with '
                             '"action" set to a tool name or "final_answer".'})
    return None


def agentic_answer(run, question):
    seen = {}   # chunk id -> chunk, for the sources panel
    tools = make_tools(run, question, seen)
    answer = gemini_agent(run, question, tools) if run.provider == "gemini" else json_agent(run, question, tools)

    sources = list(seen.values())
    if answer and answer.strip():
        run.step("Agent answered", kind="ok")
        return answer, sources

    # step limit reached (or empty reply): answer from everything the agent has gathered so far
    run.step("Step limit reached", f"answering from the {len(sources)} passages gathered", kind="warn")
    if not sources:
        sources = retrieve(run, question)      # the agent never searched: fall back to plain RAG
    sources.sort(key=lambda c: (c["year"], c["page"]))
    return generate_answer(run, question, sources), sources


# ---------------------------------------------------------------------------- entry point
def run_pipeline(mode, question, options):
    run = Run(mode, options)
    techniques = [name for name, on in (("Rerank", options.get("rerank")),
                                        ("Corrective", options.get("corrective")),
                                        ("Agentic", options.get("agentic"))) if on]
    verdict = None

    if options.get("agentic"):
        answer, sources = agentic_answer(run, question)
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
                    for c in sources],
        "llm_calls": run.llm_calls,
        "seconds": round(time.perf_counter() - run.started, 2),
        "years_asked": sorted(years_in(question)),
        "top_k": {"keyword": options.get("keyword_top_k", DEFAULT_KEYWORD_TOP_K) if mode == "hybrid" else None,
                  "vector": options.get("vector_top_k", DEFAULT_VECTOR_TOP_K),
                  "final": options.get("final_top_k", DEFAULT_FINAL_TOP_K)},
    }
