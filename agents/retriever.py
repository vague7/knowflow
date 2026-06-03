"""Hybrid retrieval agent with dense + sparse search and cross-encoder reranking."""

import os
import pickle

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ChromaDB setup
chroma_client = chromadb.Client()
collection_name = os.getenv("CHROMA_COLLECTION", "kb_docs")
collection = chroma_client.get_or_create_collection(name=collection_name)

# Load BM25 index and chunks store
INGESTION_DIR = os.path.join(os.path.dirname(__file__), "..", "ingestion")
BM25_PATH = os.path.join(INGESTION_DIR, "bm25_index.pkl")
CHUNKS_PATH = os.path.join(INGESTION_DIR, "chunks_store.pkl")

bm25_index = None
chunks_store = []

if os.path.exists(BM25_PATH) and os.path.exists(CHUNKS_PATH):
    with open(BM25_PATH, "rb") as f:
        bm25_index = pickle.load(f)
    with open(CHUNKS_PATH, "rb") as f:
        chunks_store = pickle.load(f)

# Cross-encoder for reranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def reload_indices():
    """Reload BM25 and chunks from disk (called after new uploads)."""
    global bm25_index, chunks_store
    if os.path.exists(BM25_PATH) and os.path.exists(CHUNKS_PATH):
        with open(BM25_PATH, "rb") as f:
            bm25_index = pickle.load(f)
        with open(CHUNKS_PATH, "rb") as f:
            chunks_store = pickle.load(f)


def hybrid_retrieve(query: str, k: int = 10) -> list[dict]:
    """Hybrid retrieval: dense (ChromaDB) + sparse (BM25) with RRF merge."""
    # Dense retrieval
    try:
        query_result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query",
        )
        query_vec = query_result["embedding"]
        dense_results = collection.query(query_embeddings=[query_vec], n_results=k)
        dense_docs = dense_results.get("documents", [[]])[0]
        dense_metas = dense_results.get("metadatas", [[]])[0]
        dense_ids = dense_results.get("ids", [[]])[0]
    except Exception as e:
        print(f"Dense retrieval error: {e}")
        dense_docs, dense_metas, dense_ids = [], [], []

    # Sparse retrieval (BM25)
    sparse_chunks = []
    if bm25_index and chunks_store:
        tokenized_query = query.lower().split()
        scores = bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        sparse_chunks = [chunks_store[i] for i in top_indices]

    # RRF merge (k=60)
    rrf_scores = {}
    chunk_map = {}

    # Score dense results
    for rank, (doc_id, doc, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (60 + rank)
        chunk_map[doc_id] = {"text": doc, **meta}

    # Score sparse results
    from ingestion.chunker import make_chunk_id
    for rank, chunk in enumerate(sparse_chunks):
        chunk_id = make_chunk_id(chunk["doc_title"], chunk["section_heading"],
                                 chunks_store.index(chunk))
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (60 + rank)
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = chunk

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [chunk_map[cid] for cid in sorted_ids if cid in chunk_map]


def rerank(query: str, chunks: list[dict], top_n: int = 4) -> list[dict]:
    """Rerank chunks using cross-encoder."""
    if not chunks:
        return []
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_n]]


def retrieve(query: str) -> list[dict]:
    """Public entry point: hybrid retrieve → rerank → top 4 chunks."""
    candidates = hybrid_retrieve(query, k=10)
    return rerank(query, candidates, top_n=4)
