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
EMBED_DIM = 384                                     # all-MiniLM-L6-v2 vector size

# ---- Pinecone index used by this app (override in .env) ----
# The app gets its OWN index: if it does not exist it is created automatically at startup.
PINECONE_INDEX = os.getenv("LIVE_APP_PINECONE_INDEX", "live-rag-app")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
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
        self.index_name = PINECONE_INDEX
        self.vector_count = 0
        self.needs_ingestion = True             # set by reload_from_pinecone()
        self.manifest, self.pages, self.chunks, self.chunk_by_id, self.years = [], [], [], {}, []
        self._keyword = ([], None)               # (chunks, BM25 index) - swapped together
        self._model_lock = threading.Lock()     # torch models are shared between request threads
        self._ingest_lock = threading.Lock()    # only one ingestion at a time

    # ------------------------------------------------------------------ loading
    def load(self):
        try:
            needed = ["PineCone_API_Key"]          # the index itself is created if missing
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
            # all retrieval data comes from the vector database - nothing is read from the PDFs here
            # (documents get into Pinecone only through the Data Ingestion page)
            self.status = f"loading chunks from Pinecone index '{self.index_name}'"
            self.reload_from_pinecone()
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
        """Connect to the app's own Pinecone index (LIVE_APP_PINECONE_INDEX).
        - index does not exist         -> create it (serverless, 384-dim, cosine) and wait until ready
        - index exists and fits        -> use it
        - index exists but is another kind (other dimension, sparse, or a document-schema index
          with built-in embedding)     -> stop with a clear message (the app writes its own
                                          384-dim MiniLM vectors through the vectors API)"""
        pc = Pinecone(api_key=env("PineCone_API_Key"))
        name = PINECONE_INDEX
        self.index_name = name

        if name not in pc.list_indexes().names():
            self.status = f"creating Pinecone index '{name}'"
            print(f"[KnowledgeBase] Pinecone index '{name}' not found - creating it "
                  f"({EMBED_DIM}-dim, cosine, serverless {PINECONE_CLOUD}/{PINECONE_REGION})")
            pc.create_index(name=name, dimension=EMBED_DIM, metric="cosine",
                            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION))
            for _ in range(60):                       # new indexes take a few seconds to be ready
                if pc.describe_index(name).status["ready"]:
                    break
                time.sleep(2)
            else:
                raise RuntimeError(f"Pinecone index '{name}' was created but is not ready yet - restart in a minute")
        else:
            problems = self._index_problems(pc.describe_index(name))
            if problems:
                raise RuntimeError(
                    f"Pinecone index '{name}' cannot be used by this app: {'; '.join(problems)}. "
                    f"Set LIVE_APP_PINECONE_INDEX in .env to a new name (it will be created "
                    f"automatically) or delete that index in the Pinecone console.")
        self.index = pc.Index(name)

    @staticmethod
    def _index_problems(desc):
        """Why an existing index does not fit the app ([] = it fits)."""
        d = desc.to_dict() if hasattr(desc, "to_dict") else dict(desc)
        serverless = (d.get("spec") or {}).get("serverless") or {}
        problems = []
        # Every serverless index now reports a schema; a plain vector index only has the built-in
        # fields "_values" / "_sparse_values". Document-schema indexes add their own text fields
        # (e.g. "chunk_text" with an integrated embedding model) and work with the documents API only.
        fields = ((d.get("schema") or serverless.get("schema") or {}).get("fields") or {})
        doc_fields = [f for f, spec in fields.items()
                      if not f.startswith("_") or (isinstance(spec, dict) and spec.get("embed"))]
        if doc_fields or d.get("embed"):
            problems.append(f"it is a document-schema / integrated-embedding index (fields: "
                            f"{', '.join(doc_fields) or 'embed'}; documents API only)")
        if d.get("dimension") != EMBED_DIM:
            problems.append(f"dimension is {d.get('dimension')}, the app's embeddings have {EMBED_DIM}")
        if d.get("metric") not in (None, "cosine"):
            problems.append(f"metric is {d.get('metric')}, the app needs cosine")
        if d.get("vector_type") not in (None, "dense"):
            problems.append(f"vector type is {d.get('vector_type')}, the app needs dense")
        return problems

    def count_vectors(self):
        ns = self.index.describe_index_stats().namespaces.get(NAMESPACE)
        return ns.vector_count if ns else 0

    def embed(self, texts):
        with self._model_lock:
            return self.embedder.encode(texts)

    # ------------------------------------------------------------------ data from Pinecone
    @staticmethod
    def chunk_from_metadata(chunk_id, meta):
        """One chunk, rebuilt from the metadata stored with its vector (see ingestion.chunk_metadata)."""
        return {"id": chunk_id, "year": int(meta.get("year") or 0), "page": int(meta.get("page") or 0),
                "doc_id": meta.get("doc_id", ""), "source": meta.get("source", ""),
                "label": meta.get("label", ""), "text": meta.get("text", ""),
                "chunk_index": int(meta.get("chunk_index") or 0)}

    def load_vectors(self):
        """Every vector's id + metadata in the app's namespace (list ids, then fetch in batches)."""
        ids = [item.id for page in self.index.list(namespace=NAMESPACE) for item in page]
        records = {}
        for start in range(0, len(ids), 100):
            records.update(self.index.fetch(ids=ids[start:start + 100], namespace=NAMESPACE).vectors)
        return {vid: (v.metadata or {}) for vid, v in records.items()}

    def reload_from_pinecone(self, expected=None):
        """Load ALL retrieval data from the vector database: chunk texts (for BM25, reranking and
        the LLM), full page texts (read_policy_page tool) and document headline info
        (list_policy_documents tool). Called at startup and after every ingestion."""
        vectors = self.load_vectors()
        for _ in range(10):                        # just after an ingestion Pinecone may still lag
            if expected is None or len(vectors) == expected:
                break
            time.sleep(2)
            vectors = self.load_vectors()

        chunks = [self.chunk_from_metadata(vid, m) for vid, m in vectors.items() if m.get("text")]
        chunks.sort(key=lambda c: (c["year"], c["page"], c["chunk_index"]))
        missing_text = len(vectors) - len(chunks)

        pages = sorted(({"year": int(m["year"]), "doc_id": m.get("doc_id", ""), "page": int(m["page"]),
                         "text": m["page_text"], "source": m.get("source", "")}
                        for m in vectors.values() if m.get("page_text")),
                       key=lambda p: (p["year"], p["page"]))
        docs = {}
        for m in vectors.values():                 # one entry per document (headline CL/SL/EL)
            if m.get("doc_id"):
                docs.setdefault(m["doc_id"], {"year": int(m.get("year") or 0), "doc_id": m["doc_id"],
                                              "file": m.get("source", ""),
                                              **{k: m.get(k) for k in ("effective_from", "effective_to",
                                                                       "cl", "sl", "el")}})
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None

        # swap in the new data (searches running right now keep using the old lists)
        self.manifest = sorted(docs.values(), key=lambda d: d["year"])
        self.pages = pages
        self.chunk_by_id = {c["id"]: c for c in chunks}
        self.chunks = chunks
        self._keyword = (chunks, bm25)              # BM25 is built from the text stored in Pinecone
        self.years = sorted({c["year"] for c in chunks if c["year"]})
        self.vector_count = len(vectors)
        self.needs_ingestion = not chunks or missing_text > 0
        print(f"[KnowledgeBase] loaded from Pinecone '{self.index_name}': {len(vectors)} vectors, "
              f"{len(chunks)} chunks with text, {len(pages)} pages, {len(docs)} documents"
              + (f" - {missing_text} vectors without text (run Data Ingestion)" if missing_text else ""))

    # ------------------------------------------------------------------ ingestion
    def ingest(self):
        """Leave_Policy documents -> chunks -> embeddings -> Pinecone with document / chunk hash
        checks (see ingestion.py), then reload all retrieval data from Pinecone."""
        from ingestion import run_ingestion          # imported here to avoid a circular import

        if not self._ingest_lock.acquire(blocking=False):
            raise IngestBusy("An ingestion is already running")
        self.ingesting = True
        try:
            print("[KnowledgeBase] ingestion started")
            report = run_ingestion(self)
            s = report["summary"]
            self.reload_from_pinecone(expected=s["chunks"])
            s["chunks_loaded_from_db"] = len(self.chunks)
            self.last_ingest = report
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
        result = self.index.query(vector=vector, top_k=k, namespace=NAMESPACE, include_metadata=True,
                                  filter={"year": {"$eq": year}} if year else None)
        found = []
        for m in result.matches:
            if m.id not in self.chunk_by_id and (m.metadata or {}).get("text"):
                # chunk added to Pinecone after the last reload: take it straight from its metadata
                self.chunk_by_id[m.id] = self.chunk_from_metadata(m.id, m.metadata)
            if m.id in self.chunk_by_id:
                found.append((m.id, float(m.score)))
        return found

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
