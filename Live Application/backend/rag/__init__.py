"""
RAG pipelines served to the UI.

  run.py              Run: options + timed trace (steps, LLM calls) of one pipeline run
  prompts.py          prompt texts per dataset (leave policy / bank documents)
  traditional_rag.py  Traditional RAG retrieval: vector search (Pinecone)
  hybrid_rag.py       Hybrid RAG retrieval: BM25 keyword search + vector search, fused with RRF
  reranking.py        cross-encoder reranking + before / after table
  retrieval.py        retrieve(): the section's search (+ rerank, + bank guardrail screening)
  corrective_rag.py   Corrective RAG: grade chunks, rewrite the query, search again
  AgenticRAG.py       Agentic RAG: Gemini agent with policy tools + live Wikipedia tools
  generation.py       final answer from the retrieved chunks
  pipelines.py        run_pipeline(): puts it all together (entry point used by app.py)
"""
