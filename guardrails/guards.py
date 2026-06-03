"""Basic input and output guardrails — pure functions, no LLM calls."""

import re
from datetime import datetime, timezone


def check_length(query: str) -> tuple[bool, str]:
    """Check if query is within the 500 character limit."""
    if len(query) > 500:
        return False, "Query too long. Max 500 characters."
    return True, ""


def check_injection(query: str) -> tuple[bool, str]:
    """Check for prompt injection patterns."""
    blocked = [
        "ignore instructions",
        "system prompt",
        "forget your",
        "you are now",
        "act as",
        "disregard",
    ]
    query_lower = query.lower()
    for phrase in blocked:
        if phrase in query_lower:
            return False, "This query is not allowed."
    return True, ""


def check_confidence(confidence: float) -> str | None:
    """Return a warning if confidence is below threshold."""
    if confidence < 0.5:
        return "⚠ Low confidence answer. Please verify with your team."
    return None


def check_staleness(chunks: list[dict]) -> str | None:
    """Return a warning if any chunk source is older than 90 days."""
    now = datetime.now(timezone.utc)
    for chunk in chunks:
        try:
            last_updated = datetime.fromisoformat(chunk.get("last_updated", ""))
            age_days = (now - last_updated).days
            if age_days > 90:
                return "⚠ Some sources may be outdated. Verify before relying on this."
        except (ValueError, TypeError):
            continue
    return None


def scrub_pii(text: str) -> str:
    """Replace email addresses and 10-digit phone numbers."""
    text = re.sub(r"[\w.-]+@[\w.-]+\.\w+", "[EMAIL REDACTED]", text)
    text = re.sub(r"\b\d{10}\b", "[PHONE REDACTED]", text)
    return text
