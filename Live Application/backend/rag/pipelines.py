"""
run_pipeline(): one question through one section (Traditional or Hybrid) with the options chosen
in the UI. Only orchestration lives here - every technique is in its own file:

    retrieval        rag/retrieval.py  (traditional_rag.py / hybrid_rag.py + reranking.py)
    Corrective RAG   rag/corrective_rag.py
    Agentic RAG      rag/AgenticRAG.py      employee leave policy: policy tools + live Wikipedia
                     rag/BankAgenticRAG.py  bank documents: internal tools only (no web), guard-railed
    answer           rag/generation.py
    bank guardrails  guardrails/            (input -> access filter -> screening -> output -> audit)
"""

import time

from core.utils import years_in
import guardrails
from rag.run import Run, short_label
from rag.retrieval import retrieve
from rag.corrective_rag import corrective_retrieve
from rag.generation import generate_answer
from rag.AgenticRAG import run_agent
from rag.BankAgenticRAG import run_bank_agent


def run_pipeline(mode, question, options):
    agentic = bool(options.get("agentic"))
    run = Run(mode, options)              # LLM: the Google Gemini model chosen in the UI
    techniques = [name for name, on in (("Rerank", options.get("rerank")),
                                        ("Corrective", options.get("corrective")),
                                        ("Agentic", agentic),
                                        ("Guardrails", run.corpus.guardrails)) if on]
    verdict, web_sources = None, []

    # ---- bank: input guardrails + access control
    if run.corpus.guardrails:
        guard = run.guard = guardrails.BankGuard(question, options.get("role"), options.get("customer"),
                                                 run.corpus.customers)
        ok = guard.check_input()
        flagged = [c for c in guard.checks if c["status"] != "pass"]
        run.step("Input guardrails", f"{len(guard.checks)} checks · " +
                 (", ".join(f"{c['name']}: {c['status']}" for c in flagged) or "all passed"),
                 kind="guard" if ok else "blocked")
        if not ok:
            guard.audit(mode, [], "none (blocked before retrieval)")
            run.step("Audit log", f"entry {guard.audit_id} · decision: blocked", kind="guard")
            return response(run, guardrails.REFUSALS[guard.blocked], verdict, techniques, [], [], question)
        question = guard.question                 # PII typed into the question is masked
        run.where = guard.where
        run.step("Access control", f"{guard.role_label} → {guard.access_text} "
                 "(metadata filter inside the Pinecone query and the BM25 search)", kind="guard")

    # ---- retrieve + answer
    if agentic and run.domain == "bank":      # bank agent: internal tools only, runs inside the guard
        answer, sources = run_bank_agent(run, question)
    elif agentic:                             # leave agent: policy tools + live Wikipedia
        answer, sources, web_sources = run_agent(run, question)
    elif options.get("corrective"):
        sources, verdict = corrective_retrieve(run, question)
        if not sources:
            run.step("No useful chunks", "answer not found in the documents", kind="warn")
            answer = run.prompts["not_found"]
        else:
            answer = generate_answer(run, question, sources)
    else:
        sources = retrieve(run, question)
        if not sources and run.guard:
            run.step("No accessible chunks", "nothing found within this role's access", kind="warn")
            answer = run.prompts["not_found"]
        else:
            answer = generate_answer(run, question, sources)

    # ---- bank: output guardrails + audit log
    if run.guard:
        guard = run.guard
        run.step("Retrieval guardrails", guard.retrieval_summary(), kind="guard")
        answer = guard.check_output(answer)
        out = [c for c in guard.checks if c["stage"] == "output"]
        run.step("Output guardrails", ", ".join(f"{c['name']}: {c['status']}" for c in out),
                 kind="blocked" if any(c["status"] == "blocked" for c in out) else "guard")
        guard.audit(mode, [c["doc_id"] for c in sources], run.label)
        run.step("Audit log", f"entry {guard.audit_id} → logs/bank_audit.jsonl", kind="guard")

    return response(run, answer, verdict, techniques, sources, web_sources, question)


def response(run, answer, verdict, techniques, sources, web_sources, question):
    """The JSON the UI renders."""
    return {
        "mode": run.mode,
        "domain": run.domain,
        "dataset": run.corpus.name,
        "model": run.label,
        "provider": run.provider,
        "techniques": techniques,
        "answer": answer,
        "verdict": verdict,
        "steps": run.steps,
        "reranking": run.reranks,
        "guardrails": run.guard.report() if run.guard else None,
        "sources": [{"title": short_label(c), "label": c["label"], "year": c["year"], "page": c["page"],
                     "doc_id": c.get("doc_id"), "classification": c.get("classification"),
                     "score": c.get("score"), "score_name": c.get("score_name"),
                     "grade": c.get("grade"),
                     "text": c.get("evidence") or c["text"].split(": ", 1)[-1]}
                    for c in sources]
                   + [{"kind": "web", "title": w["title"], "url": w["url"], "text": w["text"]}
                      for w in web_sources],
        "llm_calls": run.llm_calls,
        "seconds": round(time.perf_counter() - run.started, 2),
        "years_asked": sorted(years_in(question)) if run.domain == "leave" else [],
        "top_k": {"keyword": run.option("keyword_top_k") if run.mode == "hybrid" else None,
                  "vector": run.option("vector_top_k"),
                  "final": run.option("final_top_k")},
    }
