"""
Agentic RAG over the Acme Leave Policy PDFs (2015-2026).

Traditional RAG:  question -> retrieve once -> answer.
Agentic RAG:      the LLM is an *agent* with tools. It decides
                  - whether it needs to search at all,
                  - what to search for (and for which year),
                  - whether the results are good enough or it should search again,
                  - when to read a full page, or look at the document list,
                  and only then writes the final, cited answer.

The search part (pages -> chunks -> BM25 + Pinecone -> hybrid_search with RRF) is the same
as in Hybrid_RAG.ipynb, just placed inside a class and with an optional year filter.

Tools given to the agent (Gemini function calling):
    list_policy_documents()                 -> which policy PDFs exist (years, doc ids, dates)
    search_leave_policy(query, year=0)      -> hybrid search (keyword + vector, fused with RRF)
    read_policy_page(year, page)            -> full text of one page, for more context

Run:
    python AgenticRAG.py                      # interactive chat
    python AgenticRAG.py "your question"      # answer one question and exit
"""

import os
import re
import sys
import json
import time

from pypdf import PdfReader
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from google import genai
from google.genai import types

load_dotenv(find_dotenv(usecwd=True))

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Leave_Policy")
NAMESPACE = "agentic-rag"   # separate from "hybrid-rag" used by Hybrid_RAG.ipynb

STOPWORDS = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were",
             "what", "which", "how", "many", "much", "does", "do", "did", "can", "i", "my", "me",
             "it", "this", "that", "by", "with", "at", "as", "from", "per", "get", "got"}

SYSTEM_PROMPT = """You are an HR assistant for Acme Technologies India Pvt. Ltd. You answer employee
questions about the company leave policy using ONLY the policy documents, reached through your tools.

There is one leave policy per calendar year (2015-2026); entitlements change from year to year.

How to work:
1. Plan: work out which leave type(s) and which year(s) the question is about. If no year is
   given, assume the current policy (2026) and say so.
2. Retrieve: call search_leave_policy with focused queries in formal policy wording, using the
   year filter whenever a year is known. For comparisons or trends, search each year separately
   (or use list_policy_documents for the headline CL/SL/EL numbers).
3. Reflect: check whether the passages actually answer the question. If not, rephrase the query,
   try another year, or read the full page with read_policy_page. Do not stop at a weak result.
4. Answer: be concise, give exact numbers and conditions, and cite every fact as
   [HR-LV-YYYY, p.N]. If the documents do not contain the answer, say so - never guess.
"""


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


class AgenticRAG:
    """Leave-policy assistant: loads the PDFs, builds the hybrid search (BM25 + Pinecone)
    and runs a Gemini tool-calling agent over it. Keeps chat history between questions."""

    def __init__(self, llm_model="gemini-3.5-flash", max_steps=8, verbose=True):
        self.llm_model = llm_model
        self.max_steps = max_steps          # max tool-calling rounds per question (safety limit)
        self.verbose = verbose              # print the agent's tool calls
        self.history = []                   # chat memory (list of types.Content)

        # Hybrid search setup - same steps as Hybrid_RAG.ipynb
        self.read_pdfs()
        self.create_chunks()
        self.build_bm25()
        self.build_vector_index()

        # Agent setup. The google-genai SDK builds each tool's schema from its type hints
        # + docstring, so the docstrings of these methods are what the LLM reads.
        self.tools = {f.__name__: f for f in
                      (self.list_policy_documents, self.search_leave_policy, self.read_policy_page)}

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            tools=list(self.tools.values()),
            # we run the tool loop ourselves so every step is visible
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    # =======================================================================
    # PART 1 - HYBRID SEARCH (same as Hybrid_RAG.ipynb)
    # =======================================================================
    def read_pdfs(self):
        with open(os.path.join(DATA_FOLDER, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.pages = []
        for doc in self.manifest:
            reader = PdfReader(os.path.join(DATA_FOLDER, doc["file"]))
            for page_no, page in enumerate(reader.pages, start=1):
                self.pages.append({
                    "year": doc["year"],
                    "doc_id": doc["doc_id"],
                    "page": page_no,
                    "text": page.extract_text(),
                })

        print(f"Read {len(self.pages)} pages from {len(self.manifest)} PDFs")

    def create_chunks(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        self.chunks = []
        for p in self.pages:
            for i, piece in enumerate(splitter.split_text(p["text"])):
                self.chunks.append({
                    "id": f"{p['year']}-p{p['page']}-c{i}",
                    "year": p["year"],
                    "page": p["page"],
                    "text": f"Leave Policy {p['year']} ({p['doc_id']}), page {p['page']}: {piece}",
                })

        self.chunk_by_id = {c["id"]: c for c in self.chunks}
        print(f"Created {len(self.chunks)} chunks")

    def build_bm25(self):
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def build_vector_index(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        pc = Pinecone(api_key=os.getenv("PineCone_API_Key"))
        index_name = os.getenv("Pinecone_Index_Name")
        if index_name not in pc.list_indexes().names():
            pc.create_index(name=index_name, dimension=384, metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        self.index = pc.Index(index_name)
        print("Connected to Pinecone index:", index_name)

        # Upload only when the namespace is not already filled (saves time on every run)
        if self.count_vectors() == len(self.chunks):
            print(f"Namespace '{NAMESPACE}' already has {len(self.chunks)} vectors - skipping upload")
            return

        try:
            self.index.delete(delete_all=True, namespace=NAMESPACE)
        except Exception:
            pass  # namespace doesn't exist yet (first run)

        vectors = self.model.encode([c["text"] for c in self.chunks])

        records = []
        for i, c in enumerate(self.chunks):
            records.append({
                "id": c["id"],
                "values": vectors[i].tolist(),
                "metadata": {"year": c["year"]},
            })

        for start in range(0, len(records), 100):
            self.index.upsert(vectors=records[start:start + 100], namespace=NAMESPACE)

        for _ in range(30):
            if self.count_vectors() == len(self.chunks):
                break
            time.sleep(2)

        print(f"Stored {self.count_vectors()} of {len(self.chunks)} vectors in Pinecone")

    def count_vectors(self):
        ns = self.index.describe_index_stats().namespaces.get(NAMESPACE)
        return ns.vector_count if ns else 0

    def keyword_search(self, question, k=20, year=None):
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        if year:
            ranked = [(c, score) for c, score in ranked if c["year"] == year]
        return [(c["id"], score) for c, score in ranked[:k]]

    def vector_search(self, question, k=20, year=None):
        result = self.index.query(vector=self.model.encode(question).tolist(), top_k=k,
                                  namespace=NAMESPACE,
                                  filter={"year": {"$eq": year}} if year else None)
        return [(match.id, match.score) for match in result.matches]

    def hybrid_search(self, question, top_k=5, year=None):
        rrf_scores = {}
        for results in [self.keyword_search(question, year=year),
                        self.vector_search(question, year=year)]:
            for rank, (chunk_id, _score) in enumerate(results, start=1):
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (60 + rank)

        ranked = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    # =======================================================================
    # PART 2 - TOOLS THE AGENT CAN CALL
    # =======================================================================
    def list_policy_documents(self) -> dict:
        """List every leave policy document available: year, document id, effective dates
        and the headline Casual (CL), Sick (SL) and Earned (EL) leave entitlements.
        Use this to find out which years exist or for a quick overview across years."""
        return {"documents": [
            {k: d[k] for k in ("year", "doc_id", "effective_from", "effective_to", "cl", "sl", "el")}
            for d in self.manifest
        ]}

    def search_leave_policy(self, query: str, year: int = 0, top_k: int = 5) -> dict:
        """Search the company leave policy documents (one PDF per year, 2015-2026) and return
        the most relevant text passages with their year and page.

        Args:
            query: What to look for, using policy wording, e.g. "earned leave carry forward limit".
            year: Policy year to restrict the search to (e.g. 2019). Use 0 to search all years.
            top_k: Number of passages to return (1-10).
        """
        year = int(year or 0)
        years = [d["year"] for d in self.manifest]
        if year and year not in years:
            return {"error": f"No policy for {year}. Available years: {years}"}
        top_k = max(1, min(int(top_k or 5), 10))

        results = []
        for chunk_id, _score in self.hybrid_search(query, top_k=top_k, year=year):
            c = self.chunk_by_id[chunk_id]
            label, text = c["text"].split(": ", 1)
            results.append({"source": label, "year": c["year"], "page": c["page"], "text": text})
        return {"results": results}

    def read_policy_page(self, year: int, page: int) -> dict:
        """Return the full text of one page of a leave policy PDF. Use this when a search
        passage is cut off or you need the surrounding context (e.g. a whole table).

        Args:
            year: Policy year, e.g. 2022.
            page: Page number, starting at 1.
        """
        year, page = int(year), int(page)
        for p in self.pages:
            if p["year"] == year and p["page"] == page:
                return {"source": f"Leave Policy {year} ({p['doc_id']}), page {page}", "text": p["text"]}
        available = [p["page"] for p in self.pages if p["year"] == year]
        return {"error": f"Page {page} of {year} not found. Pages available for {year}: {available}"}

    # =======================================================================
    # PART 3 - THE AGENT LOOP (this is what makes it "agentic")
    # =======================================================================
    def run_tool(self, call):
        """Execute one function call requested by the model."""
        args = dict(call.args or {})
        if self.verbose:
            print(f"  -> {call.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
        try:
            return self.tools[call.name](**args)
        except Exception as e:  # report tool errors back to the model so it can recover
            return {"error": f"{type(e).__name__}: {e}"}

    def ask_gemini(self, config):
        response = self.client.models.generate_content(
            model=self.llm_model, contents=self.history, config=config)
        # keep the model's turn as-is (includes thought signatures needed for tool calling)
        self.history.append(response.candidates[0].content)
        return response

    def ask(self, question):
        """Answer one question. Earlier questions in this session are remembered."""
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

        for _step in range(self.max_steps):
            response = self.ask_gemini(self.config)
            calls = response.function_calls
            if not calls:                        # no more tools wanted -> final answer
                return response.text or "(no answer)"

            tool_parts = [types.Part.from_function_response(name=c.name, response=self.run_tool(c))
                          for c in calls]
            self.history.append(types.Content(role="user", parts=tool_parts))

        # step limit reached: ask for the best answer with what has been gathered
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(
            text="Stop searching and give your best answer from the information gathered so far.")]))
        final_config = self.config.model_copy(update={"tools": None, "automatic_function_calling": None})
        return self.ask_gemini(final_config).text or "(no answer)"

    def reset(self):
        """Forget the conversation so far."""
        self.history.clear()

    def chat(self):
        """Interactive command-line chat."""
        print("\nAgentic RAG - Leave Policy assistant. Type a question ('exit' to quit, 'reset' to clear memory).")
        print("Try: 'How did earned leave change between 2015 and 2026?' "
              "or 'Can I take leave to care for my sick father?'")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            if question.lower() == "reset":
                self.reset()
                print("Memory cleared.")
                continue
            print(f"\nAgent: {self.ask(question)}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = AgenticRAG()
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nQ: {question}")
        print(f"\nA: {agent.ask(question)}")
    else:
        agent.chat()
