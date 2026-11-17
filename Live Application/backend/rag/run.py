"""
Run: everything about ONE pipeline run - the options chosen in the UI, the dataset, the LLM to use,
and the timed trace (steps, reranking tables, LLM calls) that the UI shows.
"""

import time

from core.config import (GEMINI_MODEL, model_label, DEFAULT_KEYWORD_TOP_K, DEFAULT_VECTOR_TOP_K,
                         DEFAULT_FINAL_TOP_K)
from core.knowledge_base import kb
from core.llm import llm
from rag.prompts import PROMPTS

DEFAULTS = {"keyword_top_k": DEFAULT_KEYWORD_TOP_K, "vector_top_k": DEFAULT_VECTOR_TOP_K,
            "final_top_k": DEFAULT_FINAL_TOP_K}


class Run:
    def __init__(self, mode, options):
        self.mode = mode                          # "traditional" | "hybrid"
        self.options = options
        self.domain = options.get("domain") or "leave"
        self.corpus = kb.corpus(self.domain)      # the dataset's chunks (loaded from Pinecone)
        self.prompts = PROMPTS[self.domain]
        self.guard = None                         # guardrails.BankGuard for bank questions
        self.where = None                         # access filter (metadata) applied to every search
        # the Google Gemini model chosen in the UI (default LIVE_LLM_MODEL from .env)
        self.provider = "gemini"
        self.model = options.get("model") or GEMINI_MODEL
        self.label = model_label(self.model)
        self.steps = []
        self.reranks = []                         # before / after reranking tables, one per search
        self.llm_calls = 0
        self.started = time.perf_counter()

    def step(self, title, detail="", kind="info"):
        self.steps.append({"title": title, "detail": detail,
                           "kind": kind, "t": round(time.perf_counter() - self.started, 2)})

    def llm(self, prompt, as_json=False):
        self.llm_calls += 1
        return llm.ask(prompt, as_json=as_json, model=self.model)

    def option(self, name):
        """A UI option; top-K values fall back to the defaults in core/config.py."""
        value = self.options.get(name)
        return DEFAULTS.get(name) if value is None else value


def short_label(c):
    """Short name of a chunk's document for the UI (sources list, reranking table)."""
    if c.get("category"):                                  # bank document
        who = f" · {c['customer']}" if c.get("customer") else ""
        return f"{c['doc_id']} {c.get('doc_type', '')}{who} · p.{c['page']}"
    return f"Leave Policy {c['year']} · p.{c['page']}"
