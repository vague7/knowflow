"""SQLite-backed user memory store for interaction history and profiles."""

import os
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("MEMORY_DB_PATH", "memory/memory.db")

# Ensure directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Create tables on first import
_conn = sqlite3.connect(DB_PATH)
_conn.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        query TEXT,
        answer TEXT,
        doc_titles TEXT,
        timestamp TEXT
    )
""")
_conn.execute("""
    CREATE TABLE IF NOT EXISTS user_profile (
        user_id TEXT PRIMARY KEY,
        viewed_docs TEXT DEFAULT '',
        query_count INTEGER DEFAULT 0
    )
""")
_conn.commit()
_conn.close()


def _get_conn():
    return sqlite3.connect(DB_PATH)


def log_interaction(user_id: str, query: str, answer: str, doc_titles: list[str]) -> None:
    """Log a query-answer interaction and update user profile."""
    conn = _get_conn()
    timestamp = datetime.now(timezone.utc).isoformat()
    doc_titles_str = ",".join(doc_titles)

    conn.execute(
        "INSERT INTO interactions (user_id, query, answer, doc_titles, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, query, answer, doc_titles_str, timestamp),
    )

    # Upsert user profile
    existing = conn.execute("SELECT viewed_docs, query_count FROM user_profile WHERE user_id = ?",
                            (user_id,)).fetchone()
    if existing:
        current_docs = set(existing[0].split(",")) if existing[0] else set()
        current_docs.update(doc_titles)
        current_docs.discard("")
        conn.execute(
            "UPDATE user_profile SET viewed_docs = ?, query_count = ? WHERE user_id = ?",
            (",".join(current_docs), existing[1] + 1, user_id),
        )
    else:
        conn.execute(
            "INSERT INTO user_profile (user_id, viewed_docs, query_count) VALUES (?, ?, ?)",
            (user_id, doc_titles_str, 1),
        )

    conn.commit()
    conn.close()


def get_history(user_id: str, last_n: int = 5) -> list[dict]:
    """Return last N interactions for a user, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT query, answer FROM interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, last_n),
    ).fetchall()
    conn.close()
    return [{"query": row[0], "answer": row[1]} for row in rows]


def get_profile(user_id: str) -> dict:
    """Return user profile with viewed docs and query count."""
    conn = _get_conn()
    row = conn.execute("SELECT viewed_docs, query_count FROM user_profile WHERE user_id = ?",
                       (user_id,)).fetchone()
    conn.close()
    if row:
        docs = [d for d in row[0].split(",") if d]
        return {"viewed_docs": docs, "query_count": row[1]}
    return {"viewed_docs": [], "query_count": 0}
