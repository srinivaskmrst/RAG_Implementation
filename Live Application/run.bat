@echo off
REM Starts the RAG Playground on http://localhost:8000
cd /d "%~dp0"
python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
