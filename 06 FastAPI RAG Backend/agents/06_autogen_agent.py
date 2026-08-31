"""
Microsoft AutoGen multi-agent system for thesis/report generation.
A team of specialized agents collaborates to produce a detailed, section-wise report.

Agents:
  Planner   → breaks question into report sections
  Researcher → searches RAG documents per section
  Writer    → writes detailed prose for each section
  Critic    → reviews and improves quality
  Compiler  → assembles final structured report

pip install "autogen-agentchat" "autogen-ext[openai]"

Public interface (used by routes.py):
  MODEL_ID : str
  stream(messages) → AsyncGenerator[str, None]
"""

import os
from typing import AsyncGenerator, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools import SELECTED_MODEL_ID, rag
# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-autogen"

# ── Model client ──────────────────────────────────────────────────────────────

model_client = OpenAIChatCompletionClient(
    model=SELECTED_MODEL_ID,
    api_key=rag.config["llm"]["api_key"],
    base_url=rag.config["llm"]["base_url"],
    model_capabilities={ #  autogen-ext parameter, not supported by agent_framework.openai.
        "vision": False,
        "function_calling": True,
        "json_output": True,
    },
)

# ── RAG tools ─────────────────────────────────────────────────────────────────


async def get_filter_context(query: str) -> str:
    """Get available metadata fields and filter suggestions for a query.
    Call this first when the user mentions a company name, year, or document type."""
    return rag.get_filter_context(query)

# def search_documents(query: str) -> str:
async def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base and return relevant text chunks."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    chunks = [
        f"[{doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
        for doc in results
    ]
    return "\n\n---\n\n".join(chunks)


# ── Helpers ───────────────────────────────────────────────────────────────────


def last_user_message(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


# ── Agents ────────────────────────────────────────────────────────────────────

planner = AssistantAgent(
    name="Planner",
    description="Breaks the user's question into a structured report outline.",
    model_client=model_client,
    system_message="""
            Output 3-4 numbered section titles only, directly relevant to the question asked.
            No extra text, no descriptions.
            """,
)

researcher = AssistantAgent(
    name="Researcher",
    description="Searches the document knowledge base for facts per section.",
    model_client=model_client,
    tools=[get_filter_context, search_documents],
    system_message="""
            If the query mentions a company name, year, or document type, call get_filter_context first to get available metadata filters.
            Search the knowledge base for each section. Return only facts found in the documents.
            Numerical data: rows of metric | value | year | source.
            Other facts: bullet points with (filename, p.XX). No prose, no filler.
            """,
)

writer = AssistantAgent(
    name="Writer",
    description="Writes one focused section from research findings.",
    model_client=model_client,
    system_message="""
            Write one concise section from the research. Rules:
            - Markdown tables for numerical data.
            - Bullet points for non-numerical points only.
            - Inline citations: <sup>[N]</sup> after each fact. At the end of each section, add a numbered list where each reference is on its own line: '1. filename, p.XX'
            - Bold ALL specific numbers, percentages, dates, and monetary values using **value**. No padding, no preamble.
            """,
)

critic = AssistantAgent(
    name="Critic",
    description="Approves or flags the section with one specific fix.",
    model_client=model_client,
    system_message="""
            Reply APPROVED if the section is focused, factual, and has citations.
            Otherwise give exactly one specific fix. Nothing else.
            """,
)

compiler = AssistantAgent(
    name="Compiler",
    description="Assembles sections into the final report.",
    model_client=model_client,
    system_message="""
            Assemble sections into a final report: title, table of contents, all sections in order,
            consolidated References at the end as a numbered list, one reference per line: '1. filename, p.XX'.
            Do not rewrite any content. End with TERMINATE.
            """,
)


def _build_team() -> RoundRobinGroupChat:
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(
        max_messages=20
    )
    return RoundRobinGroupChat(
        participants=[planner, researcher, writer, critic, compiler],
        termination_condition=termination,
    )


# ── Public interface ──────────────────────────────────────────────────────────


async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    team = _build_team()
    prev_source = None
    async for event in team.run_stream(
        task=last_user_message(messages),
        cancellation_token=CancellationToken(),
    ):
        if not isinstance(event, TextMessage):
            continue
        if event.source == "Compiler":
            prev_source = None
            yield event.content.replace("TERMINATE", "").strip()
        elif event.source != prev_source and event.source != "user":
            prev_source = event.source
            yield f"\n`[{event.source} working...]`\n"
