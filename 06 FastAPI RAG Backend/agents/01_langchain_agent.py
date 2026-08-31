"""
Agent setup - LLM, agent creation, and public interface.

Public interface (used by routes.py):
  MODEL_ID                 → str
  stream(messages) → AsyncGenerator[str, None]
"""

import re
from typing import AsyncGenerator, List

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from tools import get_filter_context, rag, search_documents

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-agent"

_JSON_OBJECT = re.compile(r"^\s*\{.*\}\s*$", re.DOTALL)

SYSTEM_PROMPT = """
You are a helpful document assistant.
For complex questions — especially multi-company comparisons or multi-year analyses — break them into individual queries (one per company, one per year) and call search_documents separately for each before forming a combined answer.
If the query mentions a company name, year, or document type, call get_filter_context first to get the available metadata fields and filter suggestions.
Always call search_documents to find information before answering.
Never include raw filter data, JSON, or tool output in your final response — only use them internally to guide retrieval.
Never wrap your response in code blocks or backticks.
If no documents are found, say so honestly — never make up an answer.
Always mention the source document in your answer.
Bold all specific numbers, percentages, dates, and key financial figures using **value**.
If you include a References section, format it as a numbered list with one reference per line: '1. filename, p.XX'
"""

# ── Agent ─────────────────────────────────────────────────────────────────────

agent = create_agent(
    model=rag.llm,
    tools=[get_filter_context, search_documents],
    system_prompt=SYSTEM_PROMPT,
)

# ── Public interface ──────────────────────────────────────────────────────────

async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    """Stream the agent's response token by token, yielding plain text chunks."""
    ready_to_yield = True

    async for chunk in agent.astream(
        {"messages": messages},
        stream_mode="messages",
    ):
        token, metadata = chunk

        if getattr(token, "tool_calls", None) or getattr(token, "tool_call_chunks", None) or getattr(token, "additional_kwargs", {}).get("tool_calls"):
            ready_to_yield = False
            continue

        if isinstance(token, ToolMessage):
            ready_to_yield = True
            continue

        text = getattr(token, "content", "")
        if not text or _JSON_OBJECT.match(text):
            continue

        if ready_to_yield:
            yield text
