"""
Embedding model: turns text into vectors (SentenceTransformer all-MiniLM-L6-v2, 384 dimensions).

Used by ingestion (embed every new chunk) and by vector search (embed the question).
The model is loaded once at startup and shared by all request threads.
"""

import threading

from core.config import EMBED_MODEL


class Embedder:
    def __init__(self, model_name=EMBED_MODEL):
        from sentence_transformers import SentenceTransformer   # heavy import, only when created
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._lock = threading.Lock()            # torch models are not thread-safe

    def embed(self, texts):
        """One text -> one vector; a list of texts -> an array of vectors."""
        with self._lock:
            return self.model.encode(texts)
