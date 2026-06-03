"""Shared chunking utility for ingestion pipeline."""


def make_chunk_id(doc_title: str, section_heading: str, index: int) -> str:
    """Return a stable deterministic ID for a chunk."""
    return f"{doc_title}_{section_heading}_{index}".replace(" ", "_").lower()
