"""
Knowledge base: the app's in-memory view of the vector database.

At startup (load) it connects the embedding model, Pinecone, the reranker and the LLMs, then
loads EVERY dataset's chunks from Pinecone (reload_from_pinecone). All retrieval data - chunk
texts for BM25 / reranking / the LLM, full page texts and the document list - comes from the
vectors' metadata; the PDF folders are only read by ingestion.

    kb.corpus("leave" | "bank")   one dataset: chunks, chunk_by_id, BM25, pages, manifest, ...
    kb.ingest(domain)             run ingestion (ingestion/ingest.py) and reload from Pinecone
    kb.delete_vectors(domain)     empty one dataset's namespace
"""

import threading
import time

from rank_bm25 import BM25Okapi

from core.config import DOMAINS, PINECONE_INDEX, env
from core.utils import tokenize
from core.llm import llm

# document-level fields copied from the vector metadata into the document list
DOC_KEYS = ("effective_from", "effective_to", "cl", "sl", "el",                 # leave policy
            "customer", "customer_id", "doc_type", "category", "classification")  # bank documents


class IngestBusy(RuntimeError):
    """Raised when an ingestion / deletion is requested while another one is running."""


class Corpus:
    """One dataset = one Pinecone namespace, with all its retrieval data loaded from Pinecone."""

    def __init__(self, key):
        cfg = DOMAINS[key]
        self.key, self.name, self.folder = key, cfg["name"], cfg["folder"]
        self.namespace, self.guardrails = cfg["namespace"], cfg["guardrails"]
        self.manifest, self.pages, self.chunks, self.chunk_by_id, self.years = [], [], [], {}, []
        self.customers = []
        self.keyword = ([], None)                # (chunks, BM25 index) - swapped together
        self.vector_count = 0
        self.needs_ingestion = True
        self.last_ingest = None

    def info(self):
        return {"name": self.name, "folder": self.folder.name, "namespace": self.namespace,
                "guardrails": self.guardrails, "chunks": len(self.chunks), "vectors": self.vector_count,
                "documents": len(self.manifest), "years": self.years, "customers": self.customers,
                "needs_ingestion": self.needs_ingestion, "last_ingest": self.last_ingest}


def chunk_from_metadata(chunk_id, meta):
    """One chunk, rebuilt from the metadata stored with its vector (see chunking.chunk_metadata).
    Extra fields (bank: customer, category, classification, ...) are kept for access filters."""
    meta = meta or {}
    chunk = {k: v for k, v in meta.items() if k != "page_text"}
    chunk.update({"id": chunk_id, "year": int(meta.get("year") or 0), "page": int(meta.get("page") or 0),
                  "doc_id": meta.get("doc_id", ""), "source": meta.get("source", ""),
                  "label": meta.get("label", ""), "text": meta.get("text", ""),
                  "chunk_index": int(meta.get("chunk_index") or 0)})
    return chunk


class KnowledgeBase:
    def __init__(self):
        self.ready = False
        self.error = None
        self.status = "starting"
        self.ingesting = False
        self.index_name = PINECONE_INDEX
        self.corpora = {key: Corpus(key) for key in DOMAINS}
        self.embedder = None                    # core.embedding.Embedder
        self.store = None                       # core.vector_store.VectorStore
        self.reranker = None                    # rag.reranking.Reranker
        self._ingest_lock = threading.Lock()    # only one ingestion / deletion at a time

    def corpus(self, domain="leave"):
        if domain not in self.corpora:
            raise ValueError(f"Unknown dataset '{domain}'. Use one of: {', '.join(self.corpora)}")
        return self.corpora[domain]

    @property
    def connected(self):
        return self.store is not None and self.store.index is not None and self.embedder is not None

    # ------------------------------------------------------------------ startup
    def load(self):
        from core.embedding import Embedder
        from core.vector_store import VectorStore
        from rag.reranking import Reranker
        try:
            # Pinecone (the index itself is created if missing) + Google Gemini (the only LLM)
            missing = [k for k in ("PineCone_API_Key", "GEMINI_API_KEY") if not env(k)]
            if missing:
                raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
            self.status = "connecting to Google Gemini"
            llm.connect()

            self.status = "loading embedding model"
            self.embedder = Embedder()
            self.status = "connecting to Pinecone"
            self.store = VectorStore(self.index_name).connect(on_status=self._set_status)
            # all retrieval data comes from the vector database (documents get into Pinecone
            # only through the Data Ingestion page)
            for key, corpus in self.corpora.items():
                self.status = f"loading {corpus.name} from Pinecone index '{self.index_name}'"
                self.reload_from_pinecone(key)
            self.status = "loading reranker"
            self.reranker = Reranker()
            self.status = "ready"
            self.ready = True
            print("[KnowledgeBase] ready: " + ", ".join(
                f"{c.name} {len(c.chunks)} chunks" for c in self.corpora.values()))
        except Exception as e:  # shown in the UI
            self.error = f"{type(e).__name__}: {e}"
            self.status = "error"
            print("[KnowledgeBase] FAILED:", self.error)

    def _set_status(self, text):
        self.status = text

    # ------------------------------------------------------------------ data from Pinecone
    def reload_from_pinecone(self, domain="leave", expected=None):
        """Load ALL retrieval data of one dataset from the vector database. Called at startup and
        after every ingestion / deletion."""
        corpus = self.corpus(domain)
        vectors = self.store.load_all(corpus.namespace)
        for _ in range(10):                        # just after an ingestion Pinecone may still lag
            if expected is None or len(vectors) == expected:
                break
            time.sleep(2)
            vectors = self.store.load_all(corpus.namespace)

        chunks = [chunk_from_metadata(vid, m) for vid, m in vectors.items() if m.get("text")]
        chunks.sort(key=lambda c: (c["year"], c["doc_id"], c["page"], c["chunk_index"]))
        missing_text = len(vectors) - len(chunks)

        pages = sorted(({"year": int(m.get("year") or 0), "doc_id": m.get("doc_id", ""),
                         "page": int(m.get("page") or 0), "text": m["page_text"],
                         "source": m.get("source", "")}
                        for m in vectors.values() if m.get("page_text")),
                       key=lambda p: (p["year"], p["doc_id"], p["page"]))
        docs = {}
        for m in vectors.values():                 # one entry per document
            if m.get("doc_id"):
                docs.setdefault(m["doc_id"], {"year": int(m.get("year") or 0), "doc_id": m["doc_id"],
                                              "file": m.get("source", ""),
                                              **{k: m.get(k) for k in DOC_KEYS if m.get(k) is not None}})
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None

        # swap in the new data (searches running right now keep using the old lists)
        corpus.manifest = sorted(docs.values(), key=lambda d: (d["year"], d["doc_id"]))
        corpus.pages = pages
        corpus.chunk_by_id = {c["id"]: c for c in chunks}
        corpus.chunks = chunks
        corpus.keyword = (chunks, bm25)            # BM25 is built from the text stored in Pinecone
        corpus.years = sorted({c["year"] for c in chunks if c["year"]})
        corpus.customers = sorted({d["customer"] for d in docs.values() if d.get("customer")})
        corpus.vector_count = len(vectors)
        corpus.needs_ingestion = not chunks or missing_text > 0
        print(f"[KnowledgeBase] {corpus.name}: loaded from Pinecone '{self.index_name}' / "
              f"'{corpus.namespace}': {len(vectors)} vectors, {len(chunks)} chunks with text, "
              f"{len(pages)} pages, {len(docs)} documents"
              + (f" - {missing_text} vectors without text (run Data Ingestion)" if missing_text else ""))

    # ------------------------------------------------------------------ ingestion / deletion
    def _exclusive(self):
        if not self._ingest_lock.acquire(blocking=False):
            raise IngestBusy("Another ingestion / deletion is running - try again when it has finished")
        self.ingesting = True

    def _release(self):
        self.ingesting = False
        self._ingest_lock.release()

    def ingest(self, domain="leave"):
        """Dataset folder -> chunks -> embeddings -> Pinecone (ingestion/ingest.py), then reload."""
        from ingestion.ingest import run_ingestion   # imported here to avoid a circular import

        corpus = self.corpus(domain)
        self._exclusive()
        try:
            print(f"[KnowledgeBase] ingestion started: {corpus.name}")
            report = run_ingestion(self, domain)
            s = report["summary"]
            self.reload_from_pinecone(domain, expected=s["chunks"])
            s["chunks_loaded_from_db"] = len(corpus.chunks)
            corpus.last_ingest = report
            print(f"[KnowledgeBase] ingestion done ({corpus.name}): {s['documents']} documents, "
                  f"{s['chunks']} chunks, +{s['chunks_added']} added, {s['chunks_existing']} already "
                  f"stored, -{s['chunks_removed']} removed")
            return report
        finally:
            self._release()

    def delete_vectors(self, domain):
        """Delete EVERY vector of one dataset (its whole namespace) - the other dataset is untouched."""
        corpus = self.corpus(domain)
        self._exclusive()
        try:
            started = time.perf_counter()
            deleted = self.store.delete_namespace(corpus.namespace)
            self.reload_from_pinecone(domain, expected=0)
            corpus.last_ingest = None
            print(f"[KnowledgeBase] deleted {deleted} vectors of {corpus.name} (namespace '{corpus.namespace}')")
            return {"domain": domain, "name": corpus.name, "index": self.index_name,
                    "namespace": corpus.namespace, "deleted": deleted,
                    "vectors_in_db": corpus.vector_count, "seconds": round(time.perf_counter() - started, 2),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        finally:
            self._release()


kb = KnowledgeBase()
