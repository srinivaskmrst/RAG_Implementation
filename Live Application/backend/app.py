"""
FastAPI server for the Live Application.

    GET  /                 -> the RAG UI (frontend/index.html)
    GET  /ingest           -> the Data Ingestion page (frontend/ingest.html)
    GET  /api/status       -> is the knowledge base loaded? (+ last ingestion report)
    POST /api/ingest       -> ingest the documents (document hash + chunk hash checks)
    GET  /api/models       -> Ollama + Google Gemini models for the two model dropdowns
    POST /api/rag/{mode}   -> mode = traditional | hybrid
                              body: {"question": "...", "rerank": bool, "corrective": bool, "agentic": bool}

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

from rag_core import (kb, IngestBusy, LLM_LABEL, LLM_PROVIDER, OLLAMA_MODEL, GEMINI_MODEL,
                      SUGGESTED_OLLAMA_MODELS, default_model)
from pipelines import run_pipeline

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="RAG Live Application")


@app.on_event("startup")
def start_loading():
    # load PDFs / models / Pinecone in the background so the UI can show progress
    threading.Thread(target=kb.load, daemon=True).start()


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    rerank: bool = False
    corrective: bool = False
    agentic: bool = False
    # retrieval sizes, entered in the UI
    keyword_top_k: int = Field(10, ge=1, le=50)    # chunks from keyword search (BM25), hybrid only
    vector_top_k: int = Field(10, ge=1, le=50)     # chunks from vector search (Pinecone)
    final_top_k: int = Field(5, ge=1, le=20)       # chunks sent to the LLM after fusion / reranking
    # the ONE model picked in the UI: provider = which dropdown, model = its selected value
    provider: Literal["ollama", "gemini"] | None = None
    model: str | None = Field(None, max_length=100)


@app.get("/api/models")
def models():
    """Models for the two UI dropdowns (only one model is used per question):
       ollama : downloaded models (selectable) + suggested ones not downloaded yet
       gemini : every Gemini text model the API key can use"""
    result = {"default": {"provider": LLM_PROVIDER, "model": default_model(LLM_PROVIDER)}}

    try:
        installed = kb.installed_models()
        notes = dict(SUGGESTED_OLLAMA_MODELS)
        names = list(installed) + [n for n, _ in SUGGESTED_OLLAMA_MODELS
                                   if n not in installed and f"{n}:latest" not in installed]
        result["ollama"] = {"available": True, "default": OLLAMA_MODEL, "models": [
            {"name": n, "installed": n in installed, "size_gb": installed.get(n),
             "note": notes.get(n, notes.get(n.removesuffix(":latest"), ""))} for n in names]}
    except Exception as e:
        result["ollama"] = {"available": False, "error": str(e), "models": []}

    try:
        result["gemini"] = {"available": True, "default": GEMINI_MODEL, "models": kb.gemini_models()}
    except Exception as e:
        result["gemini"] = {"available": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "models": []}
    return result


@app.get("/api/status")
def status():
    return {
        "ready": kb.ready,
        "status": kb.status,
        "error": kb.error,
        "model": LLM_LABEL,
        "provider": LLM_PROVIDER,
        "chunks": len(kb.chunks),
        "years": kb.years,
        "ingesting": kb.ingesting,
        "last_ingest": kb.last_ingest,
    }


@app.post("/api/ingest")
def ingest():   # read documents -> chunks -> embeddings -> Pinecone, with document / chunk hash checks
    if not hasattr(kb, "index") or not hasattr(kb, "embedder"):
        raise HTTPException(503, f"Still starting up ({kb.status})")
    try:
        return kb.ingest()
    except IngestBusy as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {type(e).__name__}: {str(e)[:400]}")


@app.post("/api/rag/{mode}")
def ask(mode: str, req: AskRequest):   # plain def -> FastAPI runs it in a worker thread
    if mode not in ("traditional", "hybrid"):
        raise HTTPException(404, "mode must be 'traditional' or 'hybrid'")
    if not kb.ready:
        raise HTTPException(503, kb.error or f"Knowledge base is still loading ({kb.status})")
    provider = req.provider or LLM_PROVIDER
    if provider == "ollama" and req.model and not kb.is_installed(req.model):
        raise HTTPException(400, f"Model '{req.model}' is not downloaded. Run: ollama pull {req.model}")
    if provider == "gemini" and not kb.client:
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
        if getattr(e, "code", None) == 404 and provider == "gemini":
            raise HTTPException(400, f"Gemini model '{req.model}' is not available for this API key.")
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
