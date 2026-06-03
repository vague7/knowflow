"""Document loaders for Markdown and PDF files."""

import os
import re
from datetime import datetime, timezone

import pdfplumber


def _file_metadata(filepath: str) -> dict:
    """Extract common metadata from a file path."""
    doc_title = os.path.splitext(os.path.basename(filepath))[0]
    mtime = os.path.getmtime(filepath)
    last_updated = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return {"doc_title": doc_title, "source_file": filepath, "last_updated": last_updated}


def load_markdown(filepath: str) -> list[dict]:
    """Load a markdown file and split it into chunks by ## or ### headings."""
    meta = _file_metadata(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on lines that start with ## or ###
    sections = re.split(r"(?m)^(#{2,3}\s+.+)$", content)

    chunks = []
    current_heading = "General"
    current_text = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{2,3}\s+", part):
            # Flush previous section
            if len(current_text) >= 50:
                chunks.append({
                    "text": current_text.strip(),
                    "doc_title": meta["doc_title"],
                    "section_heading": current_heading,
                    "source_file": meta["source_file"],
                    "last_updated": meta["last_updated"],
                    "file_type": "markdown",
                })
            current_heading = re.sub(r"^#{2,3}\s+", "", part).strip()
            current_text = ""
        else:
            current_text += part + "\n"

    # Flush last section
    if len(current_text) >= 50:
        chunks.append({
            "text": current_text.strip(),
            "doc_title": meta["doc_title"],
            "section_heading": current_heading,
            "source_file": meta["source_file"],
            "last_updated": meta["last_updated"],
            "file_type": "markdown",
        })

    return chunks


def load_pdf(filepath: str) -> list[dict]:
    """Load a text-based PDF and split it into chunks by detected headings."""
    meta = _file_metadata(filepath)
    chunks = []
    current_heading = "General"
    current_lines = []

    def _is_heading(line: str) -> bool:
        stripped = line.strip()
        if len(stripped) < 4:
            return False
        if len(stripped) >= 80:
            return False
        if re.match(r"^#{2,3}\s+", stripped):
            return True
        if stripped.isupper() and len(stripped) >= 4:
            return True
        return False

    def _flush():
        text = "\n".join(current_lines).strip()
        if len(text) >= 50:
            chunks.append({
                "text": text,
                "doc_title": meta["doc_title"],
                "section_heading": current_heading,
                "source_file": meta["source_file"],
                "last_updated": meta["last_updated"],
                "file_type": "pdf",
            })

    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if not page_text:
                    continue
                for line in page_text.split("\n"):
                    if _is_heading(line):
                        _flush()
                        current_heading = re.sub(r"^#{2,3}\s+", "", line.strip()).strip()
                        current_lines = []
                    else:
                        current_lines.append(line)
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        return []

    # Flush remaining
    _flush()
    return chunks
