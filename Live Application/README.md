# RAG Playground (Live Application)

Web UI + FastAPI backend that answers questions over the Acme leave policy PDFs
(`../Leave_Policy`, 2015–2026) and shows **Traditional RAG** and **Hybrid RAG** side by side.

## Run

```bat
pip install -r ..\requirements.txt  REM the project's single requirements file
run.bat                             REM or: python -m uvicorn app:app --app-dir backend --port 8000
```

Open http://localhost:8000. Startup takes ~30–60 s (embedding + reranker models, loading the data
from Pinecone); the status pill turns green when ready. Keys are read from the project `.env`
(`PineCone_API_Key` and `GEMINI_API_KEY` - both required).

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

## Two datasets: employee leave policy + bank documents

| Dataset | Folder | Pinecone namespace (index `live-rag-app`) | Home page button |
|---|---|---|---|
| Employee leave policy | `../Leave_Policy` | `live-app-hashed` | **Process RAG – Employee leave policy** |
| Bank documents (guard-railed) | `../Bank_Documents` | `bank-docs` | **Process RAG – Bank documents** |

Folders and namespaces are configured in `DOMAINS` in `backend/core/config.py`. The **Data Ingestion**
page has one card per dataset with **Ingest** and **Delete** buttons (*Delete employee leave policy*,
*Delete bank data*). Delete removes every vector of that dataset only (its namespace).
Corrective RAG and Reranking work on both datasets. Agentic RAG uses a different agent per dataset
(leave policy: policy + live Wikipedia tools; bank: internal tools only - see below).

## Guardrails for the banking RAG (`backend/guardrails/`)

Based on the bank's own Data Privacy SOP (DOC-026). Every bank question gets a 🛡 report in the
result and an entry in `logs/bank_audit.jsonl` (git-ignored).

| Stage | Guardrail |
|---|---|
| Ingestion | PAN, Aadhaar, account / card numbers, mobile, e-mail, date of birth **masked before embedding** – raw PII never reaches Pinecone or the LLM |
| Ingestion | Chunks with prompt-injection text **quarantined**; classification label kept (missing → CONFIDENTIAL); files > 20 MB rejected |
| Input | Blocks prompt injection / jailbreak, requests for OTP / PIN / CVV / password, unmasked PII, bulk PII export, another customer's data; masks PII typed into the question |
| Access | Role chosen on the home page → metadata filter **inside the Pinecone query and the BM25 search**: Relationship manager = statements + policies, Underwriter = loan files + policies, Compliance = all, Customer = own documents only |
| Context | Retrieved chunks re-checked (access, injection) and re-masked before the LLM; guarded system prompt (sources are data, never unmask, no advice) |
| Output | Answer scanned and any PII masked; a customer's answer naming another customer is withheld |
| Audit | Time, role, customer, masked question, decision, document ids |

The role is picked in the UI for this demo; a real application would take it from the login.

## LLM: Google Gemini

Every pipeline (answers, Corrective RAG grading / rewriting, Agentic RAG) uses **Google Gemini**
through the `google-genai` API. `GEMINI_API_KEY` must be set in the project `.env`.

| Setting (`.env`) | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | – | required |
| `LIVE_LLM_MODEL` | `gemini-2.5-flash` | model selected by default in the UI |

The home page has one **Google Gemini** dropdown listing every Gemini text model your API key can
use (read live from the API, ↻ reloads it). The chosen model is sent with each question and
remembered in the browser. Temporary Gemini errors (rate limit, overload, timeout) are retried with
back-off; the free tier allows only a limited number of requests per day per model.

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

## Agentic RAG use case – Leave policy + Indian labour law (`backend/rag/AgenticRAG.py`)

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

## Agentic RAG for the bank documents (`backend/rag/BankAgenticRAG.py`)

A Google Gemini agent that plans, calls tools and checks results - with **internal tools only**
(no internet tools, so confidential data cannot leave through a tool call):

| Tool | What it does |
|---|---|
| `list_bank_documents` | documents the user's role may see |
| `search_bank_documents` | the section's retriever (+ Rerank / Corrective), limited to the role, optionally to one document |
| `read_document_page` | full PII-masked text of one page (access-checked, injection-screened) |
| `check_loan_against_policy` | reads a loan file and checks CIBIL band, FOIR and LTV against the Retail Lending Credit Policy (DOC-025) - staff only |
| `calculate_emi` | EMI / total interest arithmetic |

Guardrails around the agent: input checks before it starts (blocked questions never reach it);
role filter and access check on every tool call; denied calls reported to the agent; **grounding
guardrail** - the first step must be a tool call, and an answer citing documents no tool returned
is discarded and replaced by the normal guarded pipeline; output masking; every tool call in the
audit log and in the 🛡 report.

Example (underwriter): *"Is Ananya Sharma's home loan within the bank's LTV and FOIR limits?"* →
FOIR 34.3 % (limit 60 %) is within policy; LTV 76.7 % exceeds the 75 % limit for loans above INR 75 lakh.

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

One file per concern:

```
backend/
  app.py                        FastAPI routes: status, ingest / delete per dataset, models, RAG; serves the UI
  core/                         shared building blocks
    config.py                   ALL settings (.env values, datasets/DOMAINS, chunk size, models, top-K defaults)
    utils.py                    sha256, tokenize, years_in, parse_json, matches_filter
    embedding.py                SentenceTransformer embedder (all-MiniLM-L6-v2)
    vector_store.py             Pinecone: create/check index, list / fetch / query / upsert / delete
    llm.py                      Google Gemini client (ask, retries, model list)
    knowledge_base.py           startup + each dataset's chunks / BM25 / pages loaded from Pinecone
  ingestion/
    document_loader.py          read PDF / TXT / MD, describe leave & bank documents
    chunking.py                 splitter, chunk labels, chunk-hash ids, vector metadata
    ingest.py                   incremental ingestion (document hash + chunk hash checks)
  rag/
    traditional_rag.py          vector search (Pinecone)
    hybrid_rag.py               BM25 + vector search fused with RRF
    reranking.py                cross-encoder reranker + before / after table
    retrieval.py                retrieve(): section search + rerank + guardrail screening
    corrective_rag.py           grade chunks, verdict, rewrite query, search again
    AgenticRAG.py               leave-policy agent: policy tools + live Wikipedia tools
    BankAgenticRAG.py           bank agent: internal tools only, guard-railed, grounding check
    agent_loop.py               Gemini tool-calling loop shared by both agents
    generation.py               final answer
    prompts.py                  prompt texts per dataset
    run.py                      Run: options + timed trace of one pipeline run
    pipelines.py                run_pipeline(): orchestrates the above
  guardrails/
    pii_masking.py              PAN / Aadhaar / account / card / mobile / e-mail / DOB masking
    threat_detection.py         prompt injection, secrets, unmask & bulk requests, customer names
    access_control.py           roles -> metadata filter
    audit_log.py                logs/bank_audit.jsonl
    bank_guard.py               BankGuard: input -> retrieval -> output checks for one question
frontend/                       index.html, ingest.html, styles.css, app.js, ingest.js, common.js (no build step)
```

## Gemini quota

The free tier allows only ~20 requests per day per model. One base run uses 1 call per section;
Corrective uses 2–6; Agentic 3–10 (more with Corrective on). Set `LIVE_LLM_MODEL` in `.env` to use
another Gemini model, or use a paid key, if you hit the limit.
