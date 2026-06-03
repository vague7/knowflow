"""Streamlit chat UI for the KB QA Agent."""

import streamlit as st
import requests

BASE_URL = "http://localhost:8000"

# --- Page config ---
st.set_page_config(page_title="KB QA Agent", page_icon="📚", layout="wide")

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

# --- Sidebar ---
with st.sidebar:
    st.title("📚 KB QA Agent")

    # User ID
    user_id = st.text_input("Your name / user ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id

    st.divider()

    # Ingest section
    st.subheader("Documents")
    if st.button("Ingest sample docs", use_container_width=True):
        try:
            resp = requests.post(f"{BASE_URL}/ingest", timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Ingested {data['chunks_ingested']} chunks")
            else:
                st.error(f"Ingestion failed: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Is the backend running?")

    # Upload section
    uploaded_file = st.file_uploader("Upload PDF or Markdown", type=["pdf", "md"])
    if st.button("Upload & Ingest", use_container_width=True):
        if uploaded_file:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                resp = requests.post(f"{BASE_URL}/upload", files=files, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"{data['filename']} ingested — {data['chunks_ingested']} chunks added")
                else:
                    st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the backend running?")
        else:
            st.warning("Please select a file first")

    st.divider()

    # Profile section
    if user_id:
        try:
            resp = requests.get(f"{BASE_URL}/profile/{user_id}", timeout=10)
            if resp.status_code == 200:
                profile = resp.json()
                st.caption(f"Queries so far: {profile['query_count']}")
                if profile["viewed_docs"]:
                    st.caption("Docs you've referenced:")
                    for doc in profile["viewed_docs"]:
                        st.caption(f"  • {doc}")
        except requests.exceptions.ConnectionError:
            pass

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main chat area ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("type") == "clarification":
            # Clarification message — distinct visual style
            st.warning(f"🤔 {msg['content']}")
            st.caption("I need more information to answer accurately.")
        else:
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("citations"):
                    for c in msg["citations"]:
                        st.caption(f"📄 {c['title']} › {c['section']}")
                for w in msg.get("warnings", []):
                    st.warning(w)
                if msg.get("rewritten_query"):
                    st.caption(f'🔍 Searched as: "{msg["rewritten_query"]}"')


def _handle_response(data, user_input):
    """Handle API response — works for both /query and /clarify responses."""
    if data.get("type") == "clarification":
        # Store clarification payload for the next user message
        st.session_state.pending_clarification = {
            "original_query": data["original_query"],
            "clarification_q": data["clarification_q"],
            "retries": data["retries"],
        }
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["clarification_q"],
            "type": "clarification",
        })
    else:
        # Normal answer
        st.session_state.pending_clarification = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "citations": data.get("citations", []),
            "warnings": data.get("warnings", []),
            "rewritten_query": data.get("rewritten_query", ""),
        })


# Chat input
prompt = st.chat_input("Ask about internal docs...")
if prompt:
    if not st.session_state.user_id:
        st.warning("Please enter your name first in the sidebar")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            pending = st.session_state.pending_clarification

            if pending is not None:
                # User is replying to a clarification question
                resp = requests.post(
                    f"{BASE_URL}/clarify",
                    json={
                        "user_id": st.session_state.user_id,
                        "original_query": pending["original_query"],
                        "clarification_q": pending["clarification_q"],
                        "user_reply": prompt,
                        "retries": pending["retries"],
                    },
                    timeout=60,
                )
                # Always clear pending after sending the clarify request
                st.session_state.pending_clarification = None
            else:
                # Normal query
                resp = requests.post(
                    f"{BASE_URL}/query",
                    json={"question": prompt, "user_id": st.session_state.user_id},
                    timeout=60,
                )

            if resp.status_code == 200:
                data = resp.json()
                _handle_response(data, prompt)
            else:
                error_msg = resp.json().get("detail", resp.text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {error_msg}",
                })
        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ Cannot connect to API. Is the backend running on port 8000?",
            })

        st.rerun()
