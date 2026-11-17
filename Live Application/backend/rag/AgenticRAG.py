"""
Agentic RAG use case: Leave Policy + Indian labour-law compliance agent.

An AI agent built on Google Gemini (function calling). For every question the agent decides
on its own which knowledge source it needs, calls the tools, checks the results, and repeats
until it can answer:

  INTERNAL knowledge - RAG over the company leave policy PDFs (2015-2026)
      list_policy_documents()             headline CL / SL / EL days for every year
      search_leave_policy(query, year)    the section's retriever: vector (Traditional) or
                                          BM25 + vector (Hybrid), plus Rerank / Corrective
                                          grading when those options are switched on
      read_policy_page(year, page)        full text of one PDF page

  LIVE web application - Wikipedia, which runs on MediaWiki (open-source, GPL), through its
  free public API (no API key):
      search_wikipedia(query)             find articles, e.g. "Maternity Benefit Act India"
      read_wikipedia_article(title, focus) live article text (optionally only the paragraphs
                                          about 'focus')

Example questions the agent can only answer by combining both sources:
  "Does our 2026 maternity leave meet the Indian Maternity Benefit Act?"
  "How does our 2026 earned leave compare with the Factories Act in India?"
  "Which national holidays in India are compulsory, and how many holidays does our policy give?"

Called from rag/pipelines.py run_pipeline() when the "Agentic RAG" switch is on in the UI
(employee leave policy only).
"""

import os
import re

import httpx

from core.config import AGENT_MAX_STEPS        # tool-calling rounds per question (safety limit)
from core.knowledge_base import kb
from core.llm import llm
from core.utils import years_in
from rag.retrieval import retrieve
from rag.corrective_rag import grade_chunks
from rag.generation import generate_answer
from rag.agent_loop import run_tool_loop

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia's API policy requires a User-Agent with an app name AND a contact URL or email,
# otherwise it answers 403. Set WIKIPEDIA_USER_AGENT in .env to your own contact, e.g.
#   WIKIPEDIA_USER_AGENT=MyRAGApp/1.0 (https://github.com/<you>/<repo>; you@example.com)
WIKI_HEADERS = {"User-Agent": os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "RAGPlaygroundLearningApp/1.0 (educational demo; https://www.mediawiki.org/wiki/API:Etiquette)")}
WIKI_TIMEOUT_S = 20
WIKI_MAX_CHARS = 6000               # article text handed to the model per call

_wiki_cache = {}                    # (kind, key) -> result, saves repeated API calls

SYSTEM_PROMPT = """You are an HR compliance assistant for Acme Technologies India Pvt. Ltd.
You have two knowledge sources, reached only through your tools:

1. COMPANY LEAVE POLICY (internal documents, one policy per year 2015-2026):
   list_policy_documents, search_leave_policy, read_policy_page.
2. WIKIPEDIA (live public web data, for Indian labour law, public holidays and general facts):
   search_wikipedia, read_wikipedia_article.

How to work:
1. Plan: decide which source(s) the question needs. Questions about "our"/company rules need
   the policy; questions about the law, statutory minimums or public holidays need Wikipedia;
   compliance or comparison questions need BOTH.
2. Retrieve: for the policy, search in formal policy words and use the year filter (if no year
   is given, use 2026 and say so). For Wikipedia, first search_wikipedia, then
   read_wikipedia_article on the best title, with 'focus' set to the topic you need.
   Several tool calls in one turn are allowed - be efficient.
3. Reflect: if a result does not answer the question, rephrase, try another year or article.
4. Answer concisely with exact numbers. Keep the two sources clearly apart: cite company facts
   as [HR-LV-YYYY, p.N] and web facts as [Wikipedia: <article title>]. When comparing, say
   plainly whether the company policy meets, exceeds or falls short of the public figure.
   Add one short line that Wikipedia is general information, not legal advice.
   If neither source has the answer, say so - never guess."""


# ---------------------------------------------------------------------------- Wikipedia API
def _wiki_get(params): # Send an HTTP GET request to Wikipedia and receive JSON data.
    params = {**params, "format": "json", "formatversion": 2}
    r = httpx.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def wiki_search(query, limit=5): # Search articles
    key = ("search", query.lower().strip())
    if key not in _wiki_cache:
        data = _wiki_get({"action": "query", "list": "search", "srsearch": query, "srlimit": limit})
        _wiki_cache[key] = [{"title": r["title"],
                             "snippet": re.sub(r"<[^>]+>", "", r.get("snippet", ""))}   # strip <span> marks
                            for r in data["query"]["search"]]
    return _wiki_cache[key]


def wiki_article(title): # Get the full article text and URL
    key = ("article", title.lower().strip())
    if key not in _wiki_cache:
        data = _wiki_get({"action": "query", "prop": "extracts|info", "explaintext": 1,
                          "inprop": "url", "redirects": 1, "titles": title})
        page = data["query"]["pages"][0]
        _wiki_cache[key] = None if page.get("missing") else {
            "title": page["title"], "url": page.get("fullurl", ""), "text": page.get("extract", "")}
    return _wiki_cache[key]


def focus_text(text, focus, max_chars=WIKI_MAX_CHARS): # Keep the intro + the paragraphs that mention the focus words (so long articles fit).
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not focus:
        return "\n".join(paragraphs)[:max_chars]
    words = [w for w in re.findall(r"[a-z0-9]+", focus.lower()) if len(w) > 2]
    picked = paragraphs[:2] + [p for p in paragraphs[2:] if any(w in p.lower() for w in words)]
    return "\n".join(picked)[:max_chars]


# ---------------------------------------------------------------------------- the agent
class LeaveComplianceAgent:
    """One agent run. The policy search uses exactly the section's retriever (rag/retrieval.py)
    and the options chosen in the UI (Rerank, Corrective grading)."""

    def __init__(self, run, question):
        self.run, self.question = run, question
        self.policy = kb.corpus("leave")         # the employee leave policy dataset
        self.policy_sources = {}        # chunk id -> chunk (sources panel)
        self.web_sources = {}           # title -> {"title", "url", "text"}
        self.tools = {f.__name__: f for f in (
            self.list_policy_documents, self.search_leave_policy, self.read_policy_page,
            self.search_wikipedia, self.read_wikipedia_article)}

    # ---- internal knowledge (RAG) --------------------------------------------------------
    def list_policy_documents(self) -> dict:
        """List every company leave policy document: year, document id, effective dates and the
        headline Casual (CL), Sick (SL) and Earned (EL) leave entitlements."""
        self.run.step("Tool: list_policy_documents", "company policy overview", kind="tool")
        return {"documents": [{k: d.get(k) for k in ("year", "doc_id", "effective_from",
                                                      "effective_to", "cl", "sl", "el")}
                              for d in self.policy.manifest]}

    def search_leave_policy(self, query: str, year: int = 0) -> dict:
        """Search the COMPANY leave policy documents (one PDF per year, 2015-2026) and return
        the most relevant passages with their year and page.

        Args:
            query: What to look for, in policy wording, e.g. "maternity leave weeks".
            year: Policy year to restrict the search to (e.g. 2026). Use 0 to search all years.
        """
        try:
            year = int(year or 0)
        except (TypeError, ValueError):
            found = years_in(str(year))
            year = found.pop() if found else 0
        self.run.step("Tool: search_leave_policy", f"query=“{query}”, year={year or 'all'}", kind="tool")
        if year and year not in self.policy.years:
            return {"error": f"No policy for {year}. Available years: {self.policy.years}"}

        chunks = retrieve(self.run, str(query), year=year or None)
        if self.run.options.get("corrective"):     # self-corrective agent: keep only useful chunks
            chunks = [c for c in grade_chunks(self.run, self.question, chunks)
                      if c["grade"] != "irrelevant"]
            if not chunks:
                return {"results": [], "note": "No relevant passages. Rephrase or try another year."}
        for c in chunks:
            self.policy_sources.setdefault(c["id"], c)
        return {"results": [{"source": f"[{c['doc_id']}, p.{c['page']}]", "year": c["year"],
                             "text": c["text"].split(": ", 1)[-1]} for c in chunks]}

    def read_policy_page(self, year: int, page: int) -> dict:
        """Return the full text of one page of a company leave policy PDF, for more context.

        Args:
            year: Policy year, e.g. 2026.
            page: Page number, starting at 1.
        """
        year, page = int(year), int(page)
        self.run.step("Tool: read_policy_page", f"year={year}, page={page}", kind="tool")
        for p in self.policy.pages:
            if p["year"] == year and p["page"] == page:
                return {"source": f"[{p['doc_id']}, p.{page}]", "text": p["text"]}
        return {"error": f"Page {page} of {year} not found."}

    # ---- live web application (Wikipedia API) ---------------------------------------------
    def search_wikipedia(self, query: str) -> dict:
        """Search Wikipedia (live public web data) for articles about Indian labour law,
        statutory leave, public holidays or other general facts. Returns article titles with
        a short snippet; then call read_wikipedia_article on the best title.

        Args:
            query: Search words, e.g. "Maternity Benefit Act India" or "public holidays in India".
        """
        self.run.step("Web: search_wikipedia", f"query=“{query}” (live Wikipedia API)", kind="web")
        try:
            return {"results": wiki_search(str(query))}
        except httpx.HTTPError as e:
            return {"error": f"Wikipedia is not reachable right now: {type(e).__name__}"}

    def read_wikipedia_article(self, title: str, focus: str = "") -> dict:
        """Read the live text of one Wikipedia article.

        Args:
            title: Exact article title from search_wikipedia, e.g. "Maternity Benefit Act, 1961".
            focus: Optional topic words; only the intro and the paragraphs mentioning them are
                   returned, e.g. "weeks paid leave". Leave empty for the start of the article.
        """
        self.run.step("Web: read_wikipedia_article",
                      f"“{title}”" + (f", focus=“{focus}”" if focus else ""), kind="web")
        try:
            article = wiki_article(str(title))
        except httpx.HTTPError as e:
            return {"error": f"Wikipedia is not reachable right now: {type(e).__name__}"}
        if not article or not article["text"]:
            return {"error": f"No Wikipedia article called '{title}'. Use search_wikipedia first."}
        text = focus_text(article["text"], str(focus or ""))
        self.web_sources[article["title"]] = {"title": article["title"], "url": article["url"],
                                              "text": text[:400]}
        return {"source": f"[Wikipedia: {article['title']}]", "url": article["url"], "text": text}

    # ---- the agent loop ---------------------------------------------------------------------
    # ---- the agent loop (rag/agent_loop.py) --------------------------------------------------
    def answer(self):
        """Run the Gemini tool-calling loop. Returns (answer, policy_chunks, web_sources)."""
        run = self.run
        run.step("Agent planning", f"{run.label}: 3 company-policy tools + 2 live Wikipedia tools, "
                                   f"max {AGENT_MAX_STEPS} rounds", kind="llm")
        answer = run_tool_loop(run, SYSTEM_PROMPT, self.tools, self.question)

        policy = sorted(self.policy_sources.values(), key=lambda c: (c["year"], c["page"]))
        web = list(self.web_sources.values())
        if answer:
            run.step("Agent answered", f"{len(policy)} policy passages, {len(web)} Wikipedia articles",
                     kind="ok")
            return answer, policy, web

        # step limit reached: answer from everything gathered so far
        run.step("Step limit reached", "answering from the information gathered", kind="warn")
        if not policy and not web:
            policy = retrieve(run, self.question)
        context = policy + [{"label": f"Wikipedia: {w['title']}", "text": f"Wikipedia: {w['text']}"}
                            for w in web]
        return generate_answer(run, self.question, context), policy, web


def run_agent(run, question):
    """Entry point used by rag/pipelines.py. Returns (answer, policy_chunks, web_sources)."""
    return LeaveComplianceAgent(run, question).answer()
