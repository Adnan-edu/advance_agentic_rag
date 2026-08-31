"""
Supervisor multi-agent system with LangGraph.
A supervisor routes each query to relevant specialist agents, then synthesizes their outputs.

Specialists: financial, legal_risk, technical, summary
Supervisor decides which to call (up to 4 iterations), then synthesize → final answer.

Public interface (used by routes.py):
  MODEL_ID : str
  stream(messages) → AsyncGenerator[str, None]
"""

from typing import AsyncGenerator, Dict, List, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from tools import rag

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-supervisor"

SPECIALISTS = {
    "financial":  "revenue income profit margin financial statements cash flow",
    "legal_risk": "risk factors legal proceedings regulatory compliance liabilities",
    "technical":  "product technology research development innovation strategy",
    "summary":    "overview business strategy key highlights performance",
}

SUPERVISOR_PROMPT = """
You manage specialized document analysis agents.

Agents: financial | legal_risk | technical | summary

Query: {query}
Already called: {called}
Outputs so far: {outputs}

Which agent to call next, or FINISH if you have enough information?
Rules: do not repeat an agent; FINISH when sufficient.
Respond with one word only: financial | legal_risk | technical | summary | FINISH
"""

# ── State ─────────────────────────────────────────────────────────────────────

class State(TypedDict):
    query: str
    next_agent: str # Your supervisor would decide which agent needs to be called
    agent_outputs: Dict[str, str]
    final_answer: str
    iteration: int

# ── Nodes ─────────────────────────────────────────────────────────────────────

async def supervisor(state: State) -> State:
    called = list(state["agent_outputs"].keys())
    outputs = "\n".join(f"- {k}: {v[:100]}..." for k, v in state["agent_outputs"].items()) or "none"
    result = await rag.llm.ainvoke([HumanMessage(SUPERVISOR_PROMPT.format(query=state["query"], called=called or "none", outputs=outputs))])
    decision = result.text.strip().lower()
    next_agent = decision if decision in SPECIALISTS else "FINISH"
    return {**state, "next_agent": next_agent, "iteration": state["iteration"] + 1}


def make_specialist(name: str):
    # Look up the specialist's focus keywords from the SPECIALISTS dict (e.g. "revenue income profit..." for "financial")
    focus = SPECIALISTS[name]

    # Define the async graph node function that will be registered in the LangGraph StateGraph
    async def node(state: State) -> State:
        # Prepend the specialist's focus keywords to the user's query to bias retrieval toward relevant documents
        query = f"{focus} {state['query']}"
        # Extract structured filter metadata (company, year, doc type) from the augmented query
        filters = rag.extract_filters(query)
        # Retrieve relevant document chunks from the vector store using the augmented query and filters
        context = rag.retrieve(query, filters=filters)
        # If retrieval returned nothing, set a fallback message instead of calling the LLM
        if not context or context == "No relevant documents found.":
            output = f"No relevant {name} information found."
        else:
            # Ask the LLM to answer as a domain specialist using only the retrieved context
            result = await rag.llm.ainvoke([HumanMessage(f"You are a {name} specialist focused on: {focus}.\nAnswer using only the provided context. Bold all figures using **value**.\nNever wrap your response in code blocks or backticks.\n\nQuery: {state['query']}\n\nContext:\n{context}")])
            output = result.text
        # Merge this specialist's output into the shared agent_outputs dict and return updated state
        return {**state, "agent_outputs": {**state["agent_outputs"], name: output}}

    # Return the node function so it can be registered as a named node in the graph via graph.add_node(name, ...)
    return node


async def synthesize(state: State) -> State:
    if not state["agent_outputs"]:
        return {**state, "final_answer": "No relevant information found."}
    combined = "\n\n".join(f"{k}: {v}" for k, v in state["agent_outputs"].items())
    result = await rag.llm.ainvoke([HumanMessage(f"Synthesize these analyses into one comprehensive answer.\nBold all figures using **value**. Cite sources. Never use code blocks or backticks.\nReferences format: '1. filename, p.XX'\n\nQuery: {state['query']}\n\n{combined}")])
    return {**state, "final_answer": result.text}

# ── Routing ───────────────────────────────────────────────────────────────────

def route(state: State) -> Literal["financial", "legal_risk", "technical", "summary", "synthesize"]:
    if state["next_agent"] == "FINISH" or state["iteration"] >= 4:
        return "synthesize"
    return state["next_agent"]  # type: ignore[return-value]

# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("synthesize", synthesize)
    for name in SPECIALISTS:
        graph.add_node(name, make_specialist(name))
        graph.add_edge(name, "supervisor")
    graph.set_entry_point("supervisor")
    # Add conditional edges from the "supervisor" node.
    # The `route` function inspects state["next_agent"] and returns a string
    # (e.g. "financial", "synthesize"). This third argument is the routing map:
    # it maps the string returned by `route` to the actual graph node name.
    #
    # {n: n for n in SPECIALISTS} expands to:
    #   {"financial": "financial", "legal_risk": "legal_risk", "technical": "technical", "summary": "summary"}
    # "synthesize": "synthesize" adds the final synthesis route.
    # ** merges them into one dict:
    #   {"financial": "financial", "legal_risk": "legal_risk", "technical": "technical", "summary": "summary", "synthesize": "synthesize"}
    #
    # So if route() returns "financial", execution goes to the "financial" node;
    # if it returns "synthesize", execution goes to the "synthesize" node.
    graph.add_conditional_edges(
        "supervisor", route,
        {**{n: n for n in SPECIALISTS}, "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


graph = build_graph()

# ── Public interface ──────────────────────────────────────────────────────────

async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    async for chunk in graph.astream(
        State(query=query, next_agent="", agent_outputs={}, final_answer="", iteration=0),
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] == "messages":
            msg, metadata = chunk["data"]
            if msg.content and metadata.get("langgraph_node") == "synthesize":
                yield msg.content
