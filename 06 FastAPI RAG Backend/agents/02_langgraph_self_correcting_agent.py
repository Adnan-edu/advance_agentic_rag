"""
Self-correcting RAG agent with LangGraph.
Retrieves documents using filter context, generates an answer, and retries
with a rewritten query if no useful documents were found.

Flow: retrieve → generate → [done | rewrite → retrieve]

Public interface (used by routes.py):
  MODEL_ID : str
  stream(messages) → AsyncGenerator[str, None]
"""

from typing import AsyncGenerator, List, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from tools import rag
# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-self-correcting"
MAX_ITERATIONS = 3

SYSTEM_PROMPT = """
Answer precisely using only the provided context.
For multi-company or multi-year analyses, address each individually before forming a unified answer.
Cite the source document. Bold all specific numbers, percentages, dates, and key figures using **value**.
Never wrap your response in code blocks or backticks.
If you include a References section, format it as a numbered list: '1. filename, p.XX'
"""

# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    query: str
    current_query: str
    iteration: int
    context: str
    answer: str

# ── Nodes ─────────────────────────────────────────────────────────────────────

async def retrieve(state: State) -> State:
    filters = rag.extract_filters(state["current_query"])
    context = rag.retrieve(state["current_query"], filters=filters)
    return {**state, "context": context}


async def generate(state: State) -> State:
    if not state["context"] or state["context"] == "No relevant documents found.":
        if state["iteration"] >= MAX_ITERATIONS:
            return {**state, "answer": "The documents don't contain sufficient information to answer this question confidently."}
        return {**state, "answer": ""}
    result = await rag.llm.ainvoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(f"Context:\n{state['context']}\n\nQuestion: {state['query']}")])
    return {**state, "answer": result.content}


async def rewrite(state: State) -> State:
    result = await rag.llm.ainvoke([HumanMessage(f"The search query did not return useful results.\nOriginal question: {state['query']}\nCurrent query: {state['current_query']}\n\nWrite a better, more specific search query. Respond with the query only.")])
    return {**state, "current_query": result.content.strip(), "iteration": state["iteration"] + 1, "context": "", "answer": ""}

# ── Routing ───────────────────────────────────────────────────────────────────

def should_retry(state: State) -> Literal["rewrite", "done"]:
    if state["answer"]:
        return "done"
    return "rewrite"

# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("rewrite", rewrite)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", should_retry, {"rewrite": "rewrite", "done": END})
    graph.add_edge("rewrite", "retrieve")
    return graph.compile()


graph = build_graph()

# ── Public interface ──────────────────────────────────────────────────────────

async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    async for chunk in graph.astream(
        State(query=query, current_query=query, iteration=0, context="", answer=""),
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] == "messages":
            msg, metadata = chunk["data"]
            if msg.content and metadata.get("langgraph_node") == "generate":
                yield msg.content
