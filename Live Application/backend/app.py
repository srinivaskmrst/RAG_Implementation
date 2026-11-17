"""
FastAPI server for the Live Application.

    GET  /                 -> the RAG UI (frontend/index.html)
    GET  /ingest           -> the Data Ingestion page (frontend/ingest.html)
    GET    /api/status            -> is the knowledge base loaded? (+ per-dataset counts / last ingestion)
    POST   /api/ingest/{domain}   -> ingest one dataset (document hash + chunk hash checks;
                                     bank: + ingestion guardrails)        domain = leave | bank
    DELETE /api/vectors/{domain}  -> delete ALL vectors of one dataset (its Pinecone namespace)
    GET    /api/models            -> Google Gemini models for the model dropdown
    POST   /api/rag/{mode}        -> mode = traditional | hybrid
                                     body: {"question": "...", "domain": "leave"|"bank",
                                            "role": ..., "customer": ..., "rerank": bool, ...}

Run (from the "Live Application" folder):
    python -m uvicorn app:app --app-dir backend --port 8000
then open http://localhost:8000
"""

import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from typing import Literal

from core.config import LLM_LABEL, GEMINI_MODEL
from core.knowledge_base import kb, IngestBusy
from core.llm import llm
from rag.pipelines import run_pipeline
import guardrails

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="RAG Live Application")


@app.on_event("startup")
def start_loading():
    # load PDFs / models / Pinecone in the background so the UI can show progress
    threading.Thread(target=kb.load, daemon=True).start()


Domain = Literal["leave", "bank"]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    # which documents to run RAG on: employee leave policy or bank documents (guard-railed)
    domain: Domain = "leave"
    # bank only: who is asking (role-based access control, see guardrails.py). In this demo the
    # role is picked in the UI; a real application would take it from the login.
    role: Literal["relationship_manager", "underwriter", "compliance", "customer"] = guardrails.DEFAULT_ROLE
    customer: str | None = Field(None, max_length=100)
    rerank: bool = False
    corrective: bool = False
    agentic: bool = False
    # retrieval sizes, entered in the UI
    keyword_top_k: int = Field(10, ge=1, le=50)    # chunks from keyword search (BM25), hybrid only
    vector_top_k: int = Field(10, ge=1, le=50)     # chunks from vector search (Pinecone)
    final_top_k: int = Field(5, ge=1, le=20)       # chunks sent to the LLM after fusion / reranking
    # the Google Gemini model picked in the UI dropdown (default: LIVE_LLM_MODEL in .env)
    model: str | None = Field(None, max_length=100)


@app.get("/api/models")
def models():
    """Google Gemini text models the API key can use, for the model dropdown."""
    try:
        return {"available": True, "default": GEMINI_MODEL, "models": llm.gemini_models()}
    except Exception as e:
        return {"available": False, "default": GEMINI_MODEL, "models": [],
                "error": f"{type(e).__name__}: {str(e)[:200]}"}


@app.get("/api/status")
def status():
    return {
        "ready": kb.ready,
        "status": kb.status,
        "error": kb.error,
        "model": LLM_LABEL,
        "provider": "gemini",
        "ingesting": kb.ingesting,
        "index": kb.index_name,
        "domains": {key: c.info() for key, c in kb.corpora.items()},   # leave + bank
        "bank_roles": guardrails.roles_for_ui(),
    }


def _ensure_connected():
    if not kb.connected:
        raise HTTPException(503, kb.error or f"Still starting up ({kb.status})")


@app.post("/api/ingest/{domain}")
def ingest(domain: Domain):   # documents -> chunks -> embeddings -> Pinecone (hash checks, guardrails)
    _ensure_connected()
    try:
        return kb.ingest(domain)
    except IngestBusy as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {type(e).__name__}: {str(e)[:400]}")


@app.post("/api/ingest")
def ingest_leave():           # older URL = employee leave policy
    return ingest("leave")


@app.delete("/api/vectors/{domain}")
def delete_vectors(domain: Domain):   # delete every vector of one dataset (the other is untouched)
    _ensure_connected()
    try:
        return kb.delete_vectors(domain)
    except IngestBusy as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {type(e).__name__}: {str(e)[:400]}")


@app.post("/api/rag/{mode}")
def ask(mode: str, req: AskRequest):   # plain def -> FastAPI runs it in a worker thread
    if mode not in ("traditional", "hybrid"):
        raise HTTPException(404, "mode must be 'traditional' or 'hybrid'")
    if not kb.ready:
        raise HTTPException(503, kb.error or f"Knowledge base is still loading ({kb.status})")
    corpus = kb.corpus(req.domain)
    if not corpus.chunks:
        raise HTTPException(400, f"No {corpus.name.lower()} in the vector database (Pinecone index "
                                 f"'{kb.index_name}', namespace '{corpus.namespace}'). Open Data Ingestion "
                                 f"and click 'Ingest' for {corpus.name} first.")
    if not llm.client:
        raise HTTPException(400, "Google Gemini is not configured: add GEMINI_API_KEY to .env")
    try:
        return run_pipeline(mode, req.question.strip(), req.model_dump(exclude={"question"}))
    except Exception as e:
        text = str(e)
        if "RESOURCE_EXHAUSTED" in text or getattr(e, "code", None) == 429:
            wait = re.search(r"retry in ((?:\d+h)?(?:\d+m)?(?:\d+)?)", text)   # "16h6m40.68s" -> "16h6m40"
            wait = wait.group(1) if wait and wait.group(1) else None
            raise HTTPException(429, "Gemini quota exceeded"
                                     + (" (daily free-tier limit)" if "PerDay" in text else "")
                                     + (f". Try again in about {wait}{'' if wait[-1] in 'hm' else 's'}."
                                        if wait else ". Try again later."))
        if getattr(e, "code", None) == 404:
            raise HTTPException(400, f"Gemini model '{req.model or GEMINI_MODEL}' is not available for this API key.")
        if getattr(e, "code", None) == 503:
            raise HTTPException(503, "Gemini is overloaded right now. Please try again in a minute.")
        raise HTTPException(500, f"{type(e).__name__}: {text[:500]}")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/ingest")
def ingest_page():
    return FileResponse(FRONTEND / "ingest.html")
