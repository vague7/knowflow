"""Query rewriting agent — makes queries self-contained using conversation history."""

import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3.5-flash")


def rewrite_query(query: str, chat_history: list[dict]) -> str:
    """Rewrite a query to be self-contained using recent conversation history.

    If chat_history is empty, returns the query unchanged.
    """
    if not chat_history:
        return query

    # Format last 3 turns
    recent = chat_history[:3]
    history_text = "\n".join(
        f"Q: {turn['query']}\nA: {turn['answer']}" for turn in recent
    )

    prompt = (
        f"Given this conversation history:\n"
        f"{history_text}\n\n"
        f"Rewrite this query to be fully self-contained and specific:\n"
        f"{query}\n\n"
        f"Return only the rewritten query. No explanation. No quotes."
    )

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Query rewrite error: {e}")
        return query
