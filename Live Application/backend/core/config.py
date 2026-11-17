"""
All settings of the Live Application in one place.

Values marked (.env) can be overridden in the project .env file (Python Training/.env).
"""

import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

# backend/core/config.py -> parents: core, backend, Live Application, Python Training
APP_ROOT = Path(__file__).resolve().parents[2]          # .../Live Application
PROJECT_ROOT = Path(__file__).resolve().parents[3]      # .../Python Training

load_dotenv(find_dotenv(usecwd=True))
load_dotenv(PROJECT_ROOT / ".env")                      # the project-level .env


def env(name):
    """Read a key from .env. Some keys in the project .env have a space before '=',
    so also accept the name with a trailing space."""
    return os.getenv(name) or os.getenv(name + " ")


# ---------------------------------------------------------------------------- datasets
# Each dataset has its own folder and its own namespace in the SAME Pinecone index,
# so it is ingested, searched and deleted independently.
DOMAINS = {
    "leave": {"name": "Employee leave policy", "folder": PROJECT_ROOT / "Leave_Policy",
              "namespace": "live-app-hashed", "guardrails": False},
    "bank": {"name": "Bank documents", "folder": PROJECT_ROOT / "Bank_Documents",
             "namespace": "bank-docs", "guardrails": True},      # PII masking, access control, ...
}

# ---------------------------------------------------------------------------- chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# ---------------------------------------------------------------------------- embedding
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384                                          # all-MiniLM-L6-v2 vector size

# ---------------------------------------------------------------------------- Pinecone (.env)
# The app has its OWN index: if it does not exist it is created automatically at startup.
PINECONE_INDEX = os.getenv("LIVE_APP_PINECONE_INDEX", "live-rag-app")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# ---------------------------------------------------------------------------- reranking
RERANK_MODEL = "BAAI/bge-reranker-base"

# ---------------------------------------------------------------------------- retrieval defaults
# (the UI sends the top-K values with every question)
DEFAULT_KEYWORD_TOP_K = 10
DEFAULT_VECTOR_TOP_K = 10
DEFAULT_FINAL_TOP_K = 5
MAX_REWRITES = 2                                         # Corrective RAG query rewrites
AGENT_MAX_STEPS = 8                                      # Agentic RAG tool-calling rounds

# ---------------------------------------------------------------------------- LLM: Google Gemini (.env)
# The app uses Google Gemini only (needs GEMINI_API_KEY). The model can be changed per question
# in the UI dropdown; LIVE_LLM_MODEL is the default.
GEMINI_MODEL = os.getenv("LIVE_LLM_MODEL", "gemini-2.5-flash")
LLM_TIMEOUT_MS = 60_000                                     # Gemini request timeout
LLM_LABEL = f"{GEMINI_MODEL} (Google Gemini)"


def model_label(model):
    return f"{model or GEMINI_MODEL} (Google Gemini)"
