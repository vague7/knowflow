"""Answer generation agent with inline citations."""

import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3.5-flash")


def parse_json(text: str) -> dict:
    """Parse JSON from Gemini response, stripping markdown fences if present."""
    cleaned = re.sub(r"```json|```", "", text).strip()
    return json.loads(cleaned)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Generate an answer with citations from retrieved chunks."""
    if not chunks:
        return {"answer": "No relevant documents found.", "citations": [], "confidence": 0.0}

    # Format context
    context_parts = []
    for chunk in chunks:
        header = f"--- [{chunk['doc_title']} > {chunk['section_heading']}] ---"
        context_parts.append(f"{header}\n{chunk['text']}")
    formatted_context = "\n\n".join(context_parts)

    prompt = (
        "You are a helpful internal knowledge base assistant.\n"
        "Answer the question using ONLY the context below.\n"
        "Cite sources inline as [doc_title > section_heading].\n"
        "If the answer is not in the context say so clearly.\n"
        "Do not guess or add information not present in the context.\n\n"
        f"Context:\n{formatted_context}\n\n"
        f"Question: {query}\n\n"
        "Return a JSON object with these exact keys:\n"
        '  answer     — your answer string with inline citations\n'
        '  citations  — list of objects with keys title and section\n'
        '  confidence — float from 0.0 to 1.0'
    )

    try:
        response = model.generate_content(prompt)
        result = parse_json(response.text)
        return {
            "answer": result.get("answer", response.text),
            "citations": result.get("citations", []),
            "confidence": float(result.get("confidence", 0.5)),
        }
    except (json.JSONDecodeError, Exception) as e:
        print(f"Answer parse error: {e}")
        try:
            text = response.text
        except Exception:
            text = "Failed to generate answer."
        return {"answer": text, "citations": [], "confidence": 0.5}
