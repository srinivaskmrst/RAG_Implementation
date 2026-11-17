"""
Pinecone vector database (index live-rag-app): every read / write the app makes goes through here.

- connect()  : use the app's own index; create it if it does not exist (serverless, 384-dim,
               cosine); refuse an existing index of another kind with a clear message
- each dataset lives in its own namespace (see config.DOMAINS)
"""

import time

from core.config import (env, EMBED_DIM, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION)

BATCH = 100              # Pinecone upsert / fetch / delete batch size
MAX_IDS = 10_000         # Pinecone top_k limit when values and metadata are not returned


class VectorStore:
    def __init__(self, name=PINECONE_INDEX):
        self.name = name
        self.index = None

    # ------------------------------------------------------------------ connect / create
    def connect(self, on_status=lambda text: None):
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=env("PineCone_API_Key"))
        name = self.name
        if name not in pc.list_indexes().names():
            on_status(f"creating Pinecone index '{name}'")
            print(f"[VectorStore] Pinecone index '{name}' not found - creating it "
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
            problems = self.index_problems(pc.describe_index(name))
            if problems:
                raise RuntimeError(
                    f"Pinecone index '{name}' cannot be used by this app: {'; '.join(problems)}. "
                    f"Set LIVE_APP_PINECONE_INDEX in .env to a new name (it will be created "
                    f"automatically) or delete that index in the Pinecone console.")
        self.index = pc.Index(name)
        return self

    @staticmethod
    def index_problems(desc):
        """Why an existing index does not fit the app ([] = it fits)."""
        d = desc.to_dict() if hasattr(desc, "to_dict") else dict(desc)
        serverless = (d.get("spec") or {}).get("serverless") or {}
        problems = []
        # Every serverless index reports a schema; a plain vector index only has the built-in
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

    # ------------------------------------------------------------------ read
    def count(self, namespace):
        ns = self.index.describe_index_stats().namespaces.get(namespace)
        return ns.vector_count if ns else 0

    def load_all(self, namespace):
        """Every vector's id + metadata in a namespace (list ids, then fetch in batches)."""
        ids = [item.id for page in self.index.list(namespace=namespace) for item in page]
        records = {}
        for start in range(0, len(ids), BATCH):
            records.update(self.index.fetch(ids=ids[start:start + BATCH], namespace=namespace).vectors)
        return {vid: (v.metadata or {}) for vid, v in records.items()}

    def query(self, namespace, vector, top_k, metadata_filter=None, include_metadata=True):
        return self.index.query(vector=vector, top_k=top_k, namespace=namespace,
                                filter=metadata_filter or None, include_metadata=include_metadata).matches

    def ids_matching(self, namespace, probe, metadata_filter, top_k=MAX_IDS):
        """Ids of vectors matching a metadata filter (a query with any vector + the filter)."""
        return {m.id for m in self.query(namespace, probe, top_k, metadata_filter, include_metadata=False)}

    def existing_ids(self, namespace, ids):
        found = set()
        for start in range(0, len(ids), BATCH):
            found.update(self.index.fetch(ids=ids[start:start + BATCH], namespace=namespace).vectors.keys())
        return found

    # ------------------------------------------------------------------ write
    def upsert(self, namespace, records):
        for start in range(0, len(records), BATCH):
            self.index.upsert(vectors=records[start:start + BATCH], namespace=namespace)

    def update_metadata(self, namespace, vector_id, metadata):
        self.index.update(id=vector_id, set_metadata=metadata, namespace=namespace)

    def delete_ids(self, namespace, ids):
        ids = list(ids)
        for start in range(0, len(ids), BATCH):
            self.index.delete(ids=ids[start:start + BATCH], namespace=namespace)

    def delete_namespace(self, namespace):
        """Delete EVERY vector of one namespace and wait until Pinecone reports it empty."""
        before = self.count(namespace)
        if before:
            self.index.delete(delete_all=True, namespace=namespace)
            for _ in range(15):                       # deletion is eventually consistent
                if self.count(namespace) == 0:
                    break
                time.sleep(2)
        return before

    def wait_for_count(self, namespace, expected, tries=15):
        """Pinecone is eventually consistent: wait briefly until the count matches."""
        count = self.count(namespace)
        for _ in range(tries):
            if count == expected:
                break
            time.sleep(2)
            count = self.count(namespace)
        return count
