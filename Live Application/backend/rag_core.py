"""
Shared building blocks for every RAG pipeline in the Live Application.

Same steps as the notebooks (TraditionalRAG / Hybrid_RAG / Corrective RAG / AgenticRAG.py):
    PDFs (../../Leave_Policy) -> pages -> chunks
    keyword search  : BM25
    meaning search  : SentenceTransformer (all-MiniLM-L6-v2) + Pinecone
    hybrid search   : BM25 + Pinecone fused with Reciprocal Rank Fusion (RRF)
    reranking       : CrossEncoder BAAI/bge-reranker-base
    LLM             : open-source model through Ollama (default, runs locally)
                      or Gemini (google-genai) - set LLM_PROVIDER in .env

Everything heavy (PDF parsing, embeddings, Pinecone upload, reranker) is loaded ONCE when the
server starts and then shared by all requests.
"""

import os
import re
import json
import time
import threading
from pathlib import Path

import httpx
from dotenv import load_dotenv, find_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone, ServerlessSpec
from google import genai
from google.genai import types

load_dotenv(find_dotenv(usecwd=True))
# also look for the project-level .env (two folders up) when started from another folder
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_FOLDER = Path(__file__).resolve().parents[2] / "Leave_Policy"
NAMESPACE = "live-app-hashed"   # own namespace (hash-based ids), does not touch the notebooks' data
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"

# ---- LLM settings (override in .env) ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()      # "ollama" or "gemini"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))   # context window; Ollama's default is too small
OLLAMA_TIMEOUT_S = 600                                      # CPU inference can be slow
GEMINI_MODEL = os.getenv("LIVE_LLM_MODEL", "gemini-3.5-flash")
LLM_TIMEOUT_MS = 60_000                                     # Gemini request timeout

LLM_MODEL = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL
LLM_LABEL = f"{LLM_MODEL} ({'Ollama, local' if LLM_PROVIDER == 'ollama' else 'Gemini'})"


def model_label(model, provider=None):
    return f"{model} ({'Ollama, local' if (provider or LLM_PROVIDER) == 'ollama' else 'Google Gemini'})"


def default_model(provider):
    return OLLAMA_MODEL if provider == "ollama" else GEMINI_MODEL


# Models offered in the UI dropdown (open-source models from the Ollama library that fit a
# 12 GB RAM, CPU-only laptop). Downloaded ones can be selected; others show the pull command.
SUGGESTED_OLLAMA_MODELS = [
    ("phi3:mini", "Microsoft Phi-3 Mini 3.8B · fast"),
    ("llama3.2:3b", "Meta Llama 3.2 3B · fast"),
    ("qwen2.5:3b", "Alibaba Qwen 2.5 3B · fast"),
    ("gemma2:2b", "Google Gemma 2 2B · fastest"),
    ("llama3:8b", "Meta Llama 3 8B · better, ~2x slower"),
    ("qwen2.5:7b", "Alibaba Qwen 2.5 7B · better, ~2x slower"),
    ("mistral:7b", "Mistral 7B · better, ~2x slower"),
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

STOPWORDS = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were",
             "what", "which", "how", "many", "much", "does", "do", "did", "can", "i", "my", "me",
             "it", "this", "that", "by", "with", "at", "as", "from", "per", "get", "got"}


def env(name):
    """Read a key from .env. Some keys in the project .env have a space before '=',
    so also accept the name with a trailing space."""
    return os.getenv(name) or os.getenv(name + " ")


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def years_in(text):
    """Years named in a question, e.g. {2019}. Also finds the year in 'HR-LV-2022'."""
    return {int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text)}


def parse_json(text, default):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"[\[{].*[\]}]", text or "", re.DOTALL)  # strip stray prose / fences
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return default


class IngestBusy(RuntimeError):
    """Raised when an ingestion is requested while another one is running."""


class KnowledgeBase:
    """Holds the ingested documents and offers keyword, vector, hybrid and reranked search."""

    def __init__(self):
        self.ready = False
        self.error = None
        self.status = "starting"
        self.ingesting = False
        self.last_ingest = None
        self.manifest, self.pages, self.chunks, self.chunk_by_id, self.years = [], [], [], {}, []
        self._keyword = ([], None)               # (chunks, BM25 index) - swapped together
        self._model_lock = threading.Lock()     # torch models are shared between request threads
        self._ingest_lock = threading.Lock()    # only one ingestion at a time

    # ------------------------------------------------------------------ loading
    def load(self):
        try:
            needed = ["PineCone_API_Key", "Pinecone_Index_Name"]
            if LLM_PROVIDER == "gemini":
                needed.append("GEMINI_API_KEY")
            missing = [k for k in needed if not env(k)]
            if missing:
                raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
            if LLM_PROVIDER == "ollama":
                self.status = "checking Ollama"
                self._check_ollama()

            self.status = "loading embedding model"
            self.embedder = SentenceTransformer(EMBED_MODEL)
            self.status = "connecting to Pinecone"
            self._connect_pinecone()
            self.status = "ingesting documents (hash check)"
            self.ingest()
            self.status = "loading reranker"
            import torch
            self.reranker = CrossEncoder(RERANK_MODEL, activation_fn=torch.nn.Sigmoid())
            # Gemini is available whenever a key is set, so it can be picked in the UI
            # (timeout in ms so a stuck Gemini request cannot hang a pipeline forever)
            self.client = (genai.Client(api_key=env("GEMINI_API_KEY"),
                                        http_options=types.HttpOptions(timeout=LLM_TIMEOUT_MS))
                           if env("GEMINI_API_KEY") else None)
            if LLM_PROVIDER == "ollama":
                self.status = f"loading {OLLAMA_MODEL} into memory"
                self.ask_llm("Reply with OK.")      # warm-up: first call loads the model
            self.status = "ready"
            self.ready = True
            print(f"[KnowledgeBase] ready: {len(self.pages)} pages, {len(self.chunks)} chunks")
        except Exception as e:  # shown in the UI
            self.error = f"{type(e).__name__}: {e}"
            self.status = "error"
            print("[KnowledgeBase] FAILED:", self.error)

    def _connect_pinecone(self):
        pc = Pinecone(api_key=env("PineCone_API_Key"))
        index_name = env("Pinecone_Index_Name")
        if index_name not in pc.list_indexes().names():
            pc.create_index(name=index_name, dimension=384, metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        self.index = pc.Index(index_name)

    def count_vectors(self):
        ns = self.index.describe_index_stats().namespaces.get(NAMESPACE)
        return ns.vector_count if ns else 0

    def embed(self, texts):
        with self._model_lock:
            return self.embedder.encode(texts)

    # ------------------------------------------------------------------ ingestion
    def ingest(self):
        """Read documents -> chunks -> embeddings -> Pinecone with document / chunk hash checks
        (see ingestion.py), then swap the new chunks into the search indexes."""
        from ingestion import run_ingestion          # imported here to avoid a circular import

        if not self._ingest_lock.acquire(blocking=False):
            raise IngestBusy("An ingestion is already running")
        self.ingesting = True
        try:
            print("[KnowledgeBase] ingestion started")
            report, state = run_ingestion(self)
            chunks = state["chunks"]
            bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None
            # swap in the new data (searches running right now keep using the old lists)
            self.manifest = state["manifest"]
            self.pages = state["pages"]
            self.chunk_by_id = {c["id"]: c for c in chunks}
            self.chunks = chunks
            self._keyword = (chunks, bm25)
            self.years = sorted({c["year"] for c in chunks if c["year"]})
            self.last_ingest = report
            s = report["summary"]
            print(f"[KnowledgeBase] ingestion done: {s['documents']} documents, {s['chunks']} chunks, "
                  f"+{s['chunks_added']} added, {s['chunks_existing']} already stored, "
                  f"-{s['chunks_removed']} removed")
            return report
        finally:
            self.ingesting = False
            self._ingest_lock.release()

    # ------------------------------------------------------------------ search
    def keyword_search(self, query, k=20, year=None):
        chunks, bm25 = self._keyword
        if not bm25:
            return []
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        ranked = [(c, s) for c, s in ranked if c["id"] in self.chunk_by_id]
        if year:
            ranked = [(c, s) for c, s in ranked if c["year"] == year]
        return [(c["id"], float(s)) for c, s in ranked[:k]]

    def vector_search(self, query, k=20, year=None):
        vector = self.embed(query).tolist()
        result = self.index.query(vector=vector, top_k=k, namespace=NAMESPACE,
                                  filter={"year": {"$eq": year}} if year else None)
        return [(m.id, float(m.score)) for m in result.matches if m.id in self.chunk_by_id]

    def hybrid_search(self, query, year=None, keyword_k=20, vector_k=20):
        """Top keyword_k BM25 chunks + top vector_k Pinecone chunks, fused with Reciprocal Rank
        Fusion. Returns every unique chunk (up to keyword_k + vector_k), best first."""
        rrf = {}
        for results in (self.keyword_search(query, k=keyword_k, year=year),
                        self.vector_search(query, k=vector_k, year=year)):
            for rank, (chunk_id, _score) in enumerate(results, start=1):
                rrf[chunk_id] = rrf.get(chunk_id, 0) + 1 / (60 + rank)
        return sorted(rrf.items(), key=lambda pair: pair[1], reverse=True)

    def rerank(self, query, chunk_ids, top_k=5):
        chunk_ids = [i for i in chunk_ids if i in self.chunk_by_id]   # an ingestion may have just swapped data
        if not chunk_ids:
            return []
        with self._model_lock:
            scores = self.reranker.predict([(query, self.chunk_by_id[i]["text"]) for i in chunk_ids])
        asked = years_in(query)

        def sort_key(pair):
            chunk_id, score = pair
            wrong_year = bool(asked) and self.chunk_by_id[chunk_id]["year"] not in asked
            return (wrong_year, -score)       # asked year first, then highest score

        ranked = sorted(zip(chunk_ids, scores), key=sort_key)
        return [(i, float(s)) for i, s in ranked[:top_k]]

    # ------------------------------------------------------------------ LLM
    # ---- Google Gemini (cloud API) ----
    def gemini_models(self):
        """Gemini text models this API key can use (cached for 10 minutes). Image, audio, TTS,
        live and embedding variants are left out."""
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        cached = getattr(self, "_gemini_cache", None)
        if cached and time.time() - cached[0] < 600:
            return cached[1]
        skip = ("image", "tts", "audio", "live", "embed", "vision", "aqa", "robotics", "computer-use",
                "banana", "transcribe")
        models = []
        for m in self.client.models.list():
            name = m.name.split("/")[-1]
            if ("generateContent" in (m.supported_actions or []) and name.startswith("gemini")
                    and not any(s in name for s in skip)):
                models.append({"name": name, "display_name": m.display_name or name})
        models.sort(key=lambda m: m["name"], reverse=True)          # newest versions first
        self._gemini_cache = (time.time(), models)
        return models

    def generate(self, contents, config, retries=4, model=None):
        """Gemini call with retry + backoff for temporary errors (429 rate limit, 5xx overload)."""
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        for attempt in range(retries + 1):
            try:
                return self.client.models.generate_content(model=model or GEMINI_MODEL,
                                                           contents=contents, config=config)
            except Exception as e:
                daily_quota = "PerDay" in str(e)          # waiting a few seconds will not help
                temporary = ((getattr(e, "code", None) in (429, 500, 503, 504) and not daily_quota)
                             or "timeout" in type(e).__name__.lower())
                if attempt == retries or not temporary:
                    raise
                time.sleep(2 * 2 ** attempt)   # 2s, 4s, 8s, 16s

    # ---- Ollama (open-source models running locally) ----
    def installed_models(self):
        """Chat models downloaded in Ollama: {name: size in GB}. Embedding-only models are left out."""
        try:
            tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
        except httpx.HTTPError:
            raise RuntimeError(f"Ollama is not running at {OLLAMA_URL}. Start the Ollama app and restart.")
        return {m["name"]: round(m.get("size", 0) / 1e9, 1) for m in tags.get("models", [])
                if "embed" not in m["name"].lower()}

    def is_installed(self, model):
        names = self.installed_models()
        return model in names or f"{model}:latest" in names

    def _check_ollama(self):
        if not self.is_installed(OLLAMA_MODEL):
            raise RuntimeError(f"Model '{OLLAMA_MODEL}' is not downloaded. Run: ollama pull {OLLAMA_MODEL}")

    def chat(self, messages, as_json=False, model=None):
        """Ollama chat call. messages = [{"role": "system"|"user"|"assistant", "content": "..."}]
        model = the Ollama model chosen in the UI (default: OLLAMA_MODEL from .env)."""
        body = {
            "model": model or OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",                  # keep the model in memory between questions
            "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
        }
        if as_json:
            body["format"] = "json"              # forces the model to output valid JSON
        try:
            r = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=OLLAMA_TIMEOUT_S)
        except httpx.ConnectError:
            raise RuntimeError(f"Ollama is not running at {OLLAMA_URL}")
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json()["message"]["content"]

    def ask_llm(self, prompt, as_json=False, model=None, provider=None):
        """One LLM call. provider/model come from the UI ("ollama" or "gemini");
        defaults are LLM_PROVIDER and its model from .env."""
        if (provider or LLM_PROVIDER) == "ollama":
            return self.chat([{"role": "user", "content": prompt}], as_json=as_json, model=model)
        config = types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json" if as_json else None)
        return self.generate(prompt, config, model=model).text or ""


kb = KnowledgeBase()
