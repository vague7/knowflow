# KB QA Agent

A prototype Knowledge Base QA Agent demonstrating agentic RAG concepts — hybrid retrieval (dense + BM25), cross-encoder reranking, query rewriting with conversation history, input/output guardrails, user memory, and RAGAS-based evaluation. Built as a portfolio project.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Chat UI                         │
│                   (frontend/app.py)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP (REST)
┌──────────────────────▼──────────────────────────────────────┐
│                   FastAPI Backend                            │
│                   (api/main.py)                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │Guardrails│→ │  Rewriter │→ │ Retriever│→ │  Answer   │  │
│  │ guards.py│  │rewriter.py│  │retriever │  │ answer.py │  │
│  └──────────┘  └───────────┘  │   .py    │  └───────────┘  │
│                               └─────┬────┘                  │
│  ┌──────────┐                       │                        │
│  │  Memory  │               ┌───────▼───────┐               │
│  │ store.py │               │  ChromaDB     │               │
│  │ (SQLite) │               │  + BM25       │               │
│  └──────────┘               │  + CrossEnc.  │               │
│                             └───────────────┘               │
└─────────────────────────────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Gemini 2.0     │
              │  Flash API      │
              │  + Embeddings   │
              └─────────────────┘
```

## Known Limitations

- **Scanned/image PDFs not supported** — only text-based PDFs can be parsed (pdfplumber extracts text layers only)
- **BM25 index is in-memory** — rebuilt on each upload, not persistent across server restarts without re-ingesting
- **ChromaDB in-process** — uses `chromadb.Client()` (ephemeral), not a persistent server. Re-run ingestion after restart.
- **No authentication** — user_id is self-reported, no auth layer

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Ingest sample documents
python ingestion/ingest.py

# 4. Start the FastAPI backend (port 8000)
uvicorn api.main:app --reload

# 5. Start the Streamlit UI (port 8501) — in a separate terminal
streamlit run frontend/app.py
```

## Usage

1. Open the Streamlit UI at `http://localhost:8501`
2. Enter your name/user ID in the sidebar
3. Click "Ingest sample docs" to load the knowledge base (or use the CLI: `python ingestion/ingest.py`)
4. Ask questions about IT setup, HR policies, or developer onboarding
5. Upload additional PDF or Markdown files via the sidebar

## Upload Documents at Runtime

Use the sidebar file uploader in the Streamlit UI to upload `.pdf` or `.md` files. They are saved to `uploads/` and immediately ingested into the knowledge base.

## Run RAGAS Evaluation

```bash
python evaluation/ragas_eval.py
```

This runs 5 test cases against the pipeline and reports faithfulness, answer relevancy, and context precision scores.

## Project Structure

```
kb-qa-agent/
├── ingestion/          # Document loading, chunking, embedding, storage
│   ├── loaders.py      # Markdown + PDF loaders
│   ├── chunker.py      # Chunk ID generation
│   └── ingest.py       # Orchestrator (load → embed → store)
├── agents/             # Core agent components
│   ├── retriever.py    # Hybrid retrieval + reranking
│   ├── rewriter.py     # Query rewriting with history
│   └── answer.py       # Answer generation with citations
├── memory/
│   └── store.py        # SQLite-backed user memory
├── guardrails/
│   └── guards.py       # Input/output safety checks
├── evaluation/
│   ├── test_data.json  # 5 test Q&A cases
│   └── ragas_eval.py   # RAGAS evaluation script
├── api/
│   └── main.py         # FastAPI backend (4 endpoints)
├── frontend/
│   └── app.py          # Streamlit chat UI
├── sample_docs/        # Pre-loaded sample documents
├── uploads/            # Runtime uploads (git-ignored)
├── .env.example
├── requirements.txt
└── README.md
```
