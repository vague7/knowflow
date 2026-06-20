"""Ingestion orchestrator — load, chunk, embed, and store documents."""

import os
import pickle
import sys
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from ingestion.chunker import make_chunk_id
from ingestion.loaders import load_markdown, load_pdf

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ChromaDB client — persistent storage in project directory
chroma_client = chromadb.Client()
collection_name = os.getenv("CHROMA_COLLECTION", "kb_docs")
collection = chroma_client.get_or_create_collection(name=collection_name)

# Paths for pickled indices
BM25_PATH = os.path.join(os.path.dirname(__file__), "bm25_index.pkl")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "chunks_store.pkl")


def load_all_docs(directory: str) -> list[dict]:
    """Load all .md and .pdf files from a directory."""
    chunks = []
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return chunks

    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".md"):
            chunks.extend(load_markdown(filepath))
        elif filename.endswith(".pdf"):
            chunks.extend(load_pdf(filepath))
    return chunks


def embed_and_store(chunks: list[dict]) -> None:
    """Embed chunks with Gemini, store in ChromaDB, and build BM25 index."""
    for i, chunk in enumerate(chunks):
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=chunk["text"],
                task_type="retrieval_document",
            )
            vector = result["embedding"]

            chunk_id = make_chunk_id(chunk["doc_title"], chunk["section_heading"], i)

            collection.add(
                documents=[chunk["text"]],
                embeddings=[vector],
                metadatas=[{
                    "doc_title": chunk["doc_title"],
                    "section_heading": chunk["section_heading"],
                    "source_file": chunk["source_file"],
                    "last_updated": chunk["last_updated"],
                    "file_type": chunk["file_type"],
                }],
                ids=[chunk_id],
            )
            print(f"Embedded chunk {i + 1}/{len(chunks)}: {chunk['doc_title']} > {chunk['section_heading']}")
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")

    # Build and save BM25 index
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"BM25 index saved to {BM25_PATH}")
    print(f"Chunks store saved to {CHUNKS_PATH}")


if __name__ == "__main__":
    all_chunks = load_all_docs("sample_docs")
    if not all_chunks:
        # Try from project root
        all_chunks = load_all_docs(os.path.join(os.path.dirname(__file__), "..", "sample_docs"))
    embed_and_store(all_chunks)
    print(f"Done. Ingested {len(all_chunks)} chunks.")
