"""
Microsoft Agent Framework implementation.
Uses OpenAIChatCompletionClient pointed at Gemini's OpenAI-compatible endpoint.

Docs: https://learn.microsoft.com/en-us/agent-framework/
pip install agent-framework-openai

Public interface (used by routes.py):
  MODEL_ID : str
  stream(messages) → AsyncGenerator[str, None]
"""

from typing import AsyncGenerator, List, Optional

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatCompletionClient
from tools import SELECTED_MODEL_ID, rag

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-microsoft-agent"

SYSTEM_PROMPT = """
        You are a helpful document assistant.
        If the query mentions a company name, year, or document type, call get_filter_context first to get available metadata filters.
        Always call search_documents to find information before answering.
        If no documents are found, say so honestly — never make up an answer.
        Always mention the source document in your answer.
        Bold all specific numbers, percentages, dates, and key financial figures using **value**.
        Never wrap your response in code blocks or backticks.
        If you include a References section, format it as a numbered list with one reference per line: '1. filename, p.XX'
        """

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields and filter suggestions for a query.
    Call this first when the user mentions a company name, year, or document type."""
    return rag.get_filter_context(query)


@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base and return relevant text chunks."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )

# ── Agent ─────────────────────────────────────────────────────────────────────

model_client = OpenAIChatCompletionClient(
    model=SELECTED_MODEL_ID,
    api_key=rag.config["llm"]["api_key"],
    base_url=rag.config["llm"]["base_url"],
)

agent = Agent(
    client=model_client,
    name="RAGWireAgent",
    instructions=SYSTEM_PROMPT,
    tools=[get_filter_context, search_documents],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def last_user_message(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""

# ── Public interface ──────────────────────────────────────────────────────────

async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    async for chunk in agent.run(last_user_message(messages), stream=True):
        if chunk.text:
            yield chunk.text
