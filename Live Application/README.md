# RAG Playground (Live Application)

Web UI + FastAPI backend that answers questions over the Acme leave policy PDFs
(`../Leave_Policy`, 2015–2026) and shows **Traditional RAG** and **Hybrid RAG** side by side.

## Run

```bat
pip install -r ..\requirements.txt  REM the project's single requirements file
run.bat                             REM or: python -m uvicorn app:app --app-dir backend --port 8000
```

Open http://localhost:8000. Startup takes ~30–60 s (models, loading the data from Pinecone, the
LLM); the status pill turns green when ready. Keys are read from the project `.env`
(`PineCone_API_Key`, and `GEMINI_API_KEY` for Gemini / Agentic RAG).

## Data flow – everything comes from the Pinecone index `live-rag-app`

```
Leave_Policy/*.pdf ──(Data Ingestion page only)──► Pinecone index live-rag-app ──► all retrieval
```

* The app has its **own Pinecone index** (`LIVE_APP_PINECONE_INDEX`, default `live-rag-app`,
  namespace `live-app-hashed`). If it does not exist it is **created automatically** (serverless,
  384-dim, cosine). An existing index of another kind (other dimension, or a document-schema /
  integrated-embedding index) is rejected with a clear message.
* **Ingestion** is the only code that reads the `Leave_Policy` folder. Every vector stores all data
  retrieval needs in its metadata: chunk text, year / page / document, the document's headline
  CL/SL/EL days and (on the first chunk of each page) the full page text.
* **Retrieval** reads only from Pinecone: at startup and after each ingestion the app loads every
  vector's metadata from the index. Vector search queries Pinecone; the BM25 keyword index,
  reranking, the LLM context and the Agentic tools (`read_policy_page`, `list_policy_documents`)
  all use the text loaded from Pinecone. (Pinecone's dense index has no keyword search, so BM25
  is computed in memory – but from the text stored in Pinecone, not from the files.)
* An empty index → the status pill says "needs ingestion" and questions are refused until you
  click **Ingest documents** on the Data Ingestion page.

## LLM: open-source (Ollama) or Gemini

By default the app uses an **open-source model running locally through [Ollama](https://ollama.com)**
(no API key, no quota). Install Ollama, then download the model once:

```bat
ollama pull phi3:mini
```

Settings (all optional, in the project `.env`):

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | provider selected by default in the UI: `ollama` or `gemini` |
| `OLLAMA_MODEL` | `phi3:mini` | default Ollama model |
| `OLLAMA_NUM_CTX` | `8192` | context window given to the model |
| `OLLAMA_URL` | `http://localhost:11434` | where Ollama runs |
| `LIVE_LLM_MODEL` | `gemini-3.5-flash` | default Gemini model |
| `GEMINI_API_KEY` | – | enables the Google Gemini dropdown |

**Model selection (two dropdowns, one model):** the home page has an **Ollama** card (every chat model
downloaded in Ollama, plus suggested models greyed out until you run `ollama pull <name>` and press ↻)
and a **Google Gemini** card (every Gemini text model your API key can use, read live from the API).
Only one card is active at a time (radio button; picking a model in a dropdown activates its card).
The active provider + model are sent with each question and used for every LLM call of that run.

Ollama is unlimited but slow on a CPU (~1–2 min per answer); Gemini answers in a few seconds but the
free tier allows only a limited number of requests per day per model.

On a CPU-only laptop a local model takes roughly 20–60 s per answer, and a few minutes with
Corrective or Agentic switched on. Agentic RAG uses Gemini's native tool calling with Gemini, and a
JSON-based tool loop with Ollama models (small models make more mistakes there).

## Document ingestion (hash checks)

The **Ingest documents** button on the Data Ingestion page reads all `.pdf` / `.txt` / `.md` files in
`Leave_Policy`, splits them into chunks, embeds them and stores them in Pinecone — incrementally:

1. **Document hash** – SHA-256 of the file bytes, stored as `doc_hash` in each chunk's metadata.
   If Pinecone already has vectors with that hash, the document is **unchanged** and skipped.
2. **Chunk hash** – SHA-256 of the chunk text, used as the vector id. For a **new** or **changed**
   document only chunks whose id is not yet in Pinecone are embedded and inserted; chunks that already
   exist are kept, and chunks no longer in the document are deleted.
3. Identical copies of a file are reported as **duplicate**; vectors of files removed from the folder
   are deleted.

The page then shows the number of documents processed (new / changed / unchanged), total chunks,
chunks added / already stored / removed, the vector count in Pinecone, and a per-document table.

## Agentic RAG use case – Leave policy + Indian labour law (`backend/AgenticRAG.py`)

Switching on **Agentic RAG** runs a **Google Gemini** agent (function calling) that decides by itself
which knowledge source each question needs, calls the tools, checks the results and repeats:

| Source | Tools |
|---|---|
| Company leave policy (internal, RAG) | `list_policy_documents`, `search_leave_policy` (the section's retriever + Rerank / Corrective if on), `read_policy_page` |
| **Wikipedia – live web application** (runs on the open-source MediaWiki, public API, no key) | `search_wikipedia`, `read_wikipedia_article` |

Example: *"Does our 2026 maternity leave meet the Indian Maternity Benefit Act?"* – the agent
searches the 2026 policy (26 weeks), searches Wikipedia, reads the Maternity Benefit (Amendment)
Act article (26 weeks), and answers that the policy meets the Act, citing `[HR-LV-2026, p.3]` and
`[Wikipedia: …]` separately. Wikipedia sources appear in the Sources list with a link.

Agentic RAG always uses the model selected in the **Gemini** dropdown (needs `GEMINI_API_KEY`).
Wikimedia requires a User-Agent with contact details; set your own in `.env` (optional):
`WIKIPEDIA_USER_AGENT=MyRAGApp/1.0 (https://github.com/<you>/<repo>; you@example.com)`

## What each section does

| Section | Retrieval |
|---|---|
| Traditional RAG | Vector search (MiniLM embeddings in Pinecone), top 5 → LLM |
| Hybrid RAG | BM25 + vector search fused with Reciprocal Rank Fusion, top 5 → LLM |

Optional techniques (toggles; applied to **both** sections so you can compare):

| Option | Logic |
|---|---|
| Reranking | Take 20 candidates, re-score with `BAAI/bge-reranker-base`, keep best 5 (asked year first) |
| Corrective RAG | The LLM grades each chunk relevant / partial / irrelevant; if none is relevant, rewrite the query and search again (max 2 rewrites); answer only from useful chunks |
| Agentic RAG | LLM agent with tools (`list_policy_documents`, `search_leave_policy`, `read_policy_page`) plans its own searches; its search tool uses the section's retriever, plus reranking and grading when those are switched on |

Every result shows the answer, a timed pipeline trace, the sources (with score / grade),
the run time and the number of LLM calls.

## Files

```
backend/ingestion.py   documents -> chunks -> embeddings -> Pinecone with document / chunk hash checks
backend/rag_core.py    knowledge base: BM25, Pinecone, hybrid search, reranker, LLM calls (Ollama / Gemini)
backend/pipelines.py   traditional / hybrid + rerank, corrective, agentic logic
backend/app.py         FastAPI: GET /api/status, POST /api/ingest, POST /api/rag/{traditional|hybrid}, serves the UI
frontend/              index.html, styles.css, app.js (no build step)
```

## Gemini quota (only with LLM_PROVIDER=gemini)

The free tier allows only ~20 requests per day per model. One base run uses 1 call per section;
Corrective uses 2–6; Agentic 3–10 (more with Corrective on). Set `LIVE_LLM_MODEL` in `.env` to use
another Gemini model, or use a paid key, if you hit the limit.
