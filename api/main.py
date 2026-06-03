"""FastAPI backend — 5 endpoints for the KB QA Agent."""

import os
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.retriever import reload_indices
from graph import run_query
from guardrails.guards import (
    check_confidence,
    check_injection,
    check_length,
    check_staleness,
    scrub_pii,
)
from ingestion.ingest import embed_and_store, load_all_docs
from ingestion.loaders import load_markdown, load_pdf
from memory.store import get_history, get_profile, log_interaction

app = FastAPI(title="KB QA Agent API")

# CORS — allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    user_id: str


@app.post("/ingest")
def ingest_docs():
    """Ingest sample documents and any uploaded files into the knowledge base."""
    project_root = os.path.join(os.path.dirname(__file__), "..")
    sample_dir = os.path.join(project_root, "sample_docs")
    uploads_dir = os.path.join(project_root, "uploads")

    chunks = load_all_docs(sample_dir)
    chunks.extend(load_all_docs(uploads_dir))

    if not chunks:
        raise HTTPException(status_code=400, detail="No documents found")
    embed_and_store(chunks)
    reload_indices()
    return {"status": "ok", "chunks_ingested": len(chunks)}


@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload and ingest a PDF or Markdown file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".md"):
        raise HTTPException(status_code=400, detail="Only .pdf and .md files are allowed")

    # Save file to uploads/
    filepath = os.path.join(UPLOADS_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load and ingest
    if ext == ".pdf":
        chunks = load_pdf(filepath)
    else:
        chunks = load_markdown(filepath)

    if not chunks:
        raise HTTPException(status_code=400, detail="No content could be extracted from the file")

    embed_and_store(chunks)
    reload_indices()

    return {"status": "ok", "filename": file.filename, "chunks_ingested": len(chunks)}


@app.post("/query")
async def query(body: QueryRequest):
    """Agentic QA pipeline: guardrails → LangGraph (rewrite → retrieve → generate → clarify loop)."""
    # Input guardrails
    ok, msg = check_length(body.question)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    ok, msg = check_injection(body.question)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    history = get_history(body.user_id, last_n=5)

    # Run LangGraph pipeline
    result = run_query(
        user_id=body.user_id,
        question=body.question,
        chat_history=history,
        retries=0,
    )

    # If agent needs clarification
    if result["awaiting_user"]:
        return {
            "type": "clarification",
            "clarification_q": result["clarification_q"],
            "original_query": body.question,
            "retries": result["retries"],
        }

    # Normal answer path — output guardrails
    answer = result["answer"]
    chunks = result["chunks"]
    conf_warning = check_confidence(answer["confidence"])
    stale_warning = check_staleness(chunks)
    answer["answer"] = scrub_pii(answer["answer"])

    log_interaction(
        body.user_id, body.question,
        answer["answer"],
        [c["doc_title"] for c in chunks],
    )

    return {
        "type": "answer",
        "answer": answer["answer"],
        "citations": answer["citations"],
        "confidence": answer["confidence"],
        "rewritten_query": result["current_query"],
        "warnings": [w for w in [conf_warning, stale_warning] if w is not None],
    }


class ClarifyRequest(BaseModel):
    user_id: str
    original_query: str
    clarification_q: str
    user_reply: str
    retries: int


@app.post("/clarify")
async def clarify(body: ClarifyRequest):
    """Handle the user's reply to a clarification question."""
    # Input guardrails on the reply
    ok, msg = check_length(body.user_reply)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    ok, msg = check_injection(body.user_reply)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    # Merge original query + clarification into enriched query
    enriched_query = (
        f"{body.original_query} "
        f"(Additional context: {body.clarification_q} "
        f"{body.user_reply})"
    )

    history = get_history(body.user_id, last_n=5)

    # Re-run graph with enriched query and carried retries
    result = run_query(
        user_id=body.user_id,
        question=enriched_query,
        chat_history=history,
        retries=body.retries,
    )

    # Same response logic as /query
    if result["awaiting_user"]:
        return {
            "type": "clarification",
            "clarification_q": result["clarification_q"],
            "original_query": body.original_query,
            "retries": result["retries"],
        }

    answer = result["answer"]
    chunks = result["chunks"]
    conf_warning = check_confidence(answer["confidence"])
    stale_warning = check_staleness(chunks)
    answer["answer"] = scrub_pii(answer["answer"])

    # Log with original query so memory stays clean
    log_interaction(
        body.user_id, body.original_query,
        answer["answer"],
        [c["doc_title"] for c in chunks],
    )

    # If forced END due to max retries, add a warning
    warnings = [w for w in [conf_warning, stale_warning] if w is not None]
    if body.retries >= 2:
        warnings.append(
            "⚠ Max clarifications reached. "
            "This is the best available answer — verify with your team."
        )

    return {
        "type": "answer",
        "answer": answer["answer"],
        "citations": answer["citations"],
        "confidence": answer["confidence"],
        "rewritten_query": result["current_query"],
        "warnings": warnings,
    }


@app.get("/profile/{user_id}")
def user_profile(user_id: str):
    """Return user profile with viewed docs and query count."""
    return get_profile(user_id)
