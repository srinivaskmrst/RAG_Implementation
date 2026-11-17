"""
Ingestion: documents -> chunks -> embeddings -> Pinecone.

  document_loader.py  read PDF / TXT / MD files and describe each document (leave / bank)
  chunking.py         split pages into chunks, chunk labels, chunk hash ids, vector metadata
  ingest.py           incremental ingestion with document-hash / chunk-hash checks
                      (+ bank guardrails: PII masking, injection quarantine)
Embedding (core/embedding.py) and Pinecone (core/vector_store.py) are shared with retrieval.
"""
