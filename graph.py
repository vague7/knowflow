"""LangGraph state machine — agentic clarification loop for the KB QA pipeline."""

import os
from typing import TypedDict

import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.answer import generate_answer
from agents.retriever import retrieve
from agents.rewriter import rewrite_query

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

clarify_model = genai.GenerativeModel("gemini-2.0-flash")


# ── State schema ───────────────────────────────────────────────
class AgentState(TypedDict):
    user_id: str
    original_query: str          # never mutated — always the raw input
    current_query: str           # mutated on each loop iteration
    chat_history: list[dict]     # {query, answer} from memory
    chunks: list[dict]           # retrieved chunks
    answer: dict                 # {answer, citations, confidence}
    clarification_q: str | None  # question sent back to user
    awaiting_user: bool          # True when waiting for clarification reply
    retries: int                 # increments each clarification loop


# ── Node: rewrite ──────────────────────────────────────────────
def node_rewrite(state: AgentState) -> AgentState:
    """Rewrite the current query using conversation history."""
    rewritten = rewrite_query(state["current_query"], state["chat_history"])
    return {**state, "current_query": rewritten}


# ── Node: retrieve ─────────────────────────────────────────────
def node_retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant chunks for the current query."""
    chunks = retrieve(state["current_query"])
    return {**state, "chunks": chunks}


# ── Node: generate ─────────────────────────────────────────────
def node_generate(state: AgentState) -> AgentState:
    """Generate an answer from retrieved chunks."""
    answer = generate_answer(state["current_query"], state["chunks"])
    return {**state, "answer": answer}


# ── Node: clarify ──────────────────────────────────────────────
def node_clarify(state: AgentState) -> AgentState:
    """Generate a clarifying question to send back to the user."""
    prompt = (
        "You are a helpful knowledge base assistant. You attempted to "
        "answer this question but lacked sufficient information:\n\n"
        f"Question: {state['current_query']}\n"
        f"Your answer attempt: {state['answer']['answer']}\n"
        f"Confidence: {state['answer']['confidence']}\n\n"
        "Generate a single short clarifying question to ask the user "
        "that would help you give a better answer. The question should "
        "be specific, not generic. Return only the question, no preamble."
    )

    try:
        response = clarify_model.generate_content(prompt)
        clarification_q = response.text.strip()
    except Exception as e:
        print(f"Clarification generation error: {e}")
        clarification_q = "Could you provide more details about your question?"

    return {
        **state,
        "clarification_q": clarification_q,
        "awaiting_user": True,
        "retries": state["retries"] + 1,
    }


# ── Conditional edge function ──────────────────────────────────
def should_clarify(state: AgentState) -> str:
    """Decide whether to clarify or end after generation."""
    if state["retries"] >= 2:
        return "end"

    if state["answer"]["confidence"] >= 0.6:
        return "end"

    return "clarify"


# ── Build and compile the graph ────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("rewrite", node_rewrite)
builder.add_node("retrieve", node_retrieve)
builder.add_node("generate", node_generate)
builder.add_node("clarify", node_clarify)

builder.set_entry_point("rewrite")

builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_conditional_edges(
    "generate",
    should_clarify,
    {"end": END, "clarify": "clarify"},
)
builder.add_edge("clarify", END)

graph = builder.compile()


# ── Public interface ───────────────────────────────────────────
def run_query(
    user_id: str,
    question: str,
    chat_history: list[dict],
    retries: int = 0,
) -> AgentState:
    """Run the LangGraph pipeline and return the final state."""
    initial_state = AgentState(
        user_id=user_id,
        original_query=question,
        current_query=question,
        chat_history=chat_history,
        chunks=[],
        answer={},
        clarification_q=None,
        awaiting_user=False,
        retries=retries,
    )
    return graph.invoke(initial_state)
