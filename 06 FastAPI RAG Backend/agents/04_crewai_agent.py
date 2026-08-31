"""
CrewAI agent implementation.

Public interface (same contract as all agent files):
  MODEL_ID : str
  stream(messages) -> AsyncGenerator[str, None]
"""

import re
from typing import AsyncGenerator, List, Optional

from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool as crewai_tool
import asyncio
import logging
from tools import SELECTED_MODEL_ID, rag

logger = logging.getLogger(__name__)


# -- Constants -----------------------------------------------------------------

MODEL_ID = "ragwire-crewai"

# -- LLM -----------------------------------------------------------------------
# NOTE: The model must support tool/function calling for CrewAI agents to work.
# Free models (e.g. nvidia/nemotron-3-super-120b-a12b:free) may lack tool calling
# and will output JSON action traces instead of calling tools.
# Models known to work: google/gemini-2.0-flash-001, anthropic/claude-3.5-sonnet,
# openai/gpt-4o, meta-llama/llama-3.1-70b-instruct, etc.

llm = LLM(
    model=f"openrouter/{SELECTED_MODEL_ID}",
    api_key=rag.config["llm"]["api_key"],
    base_url=rag.config["llm"]["base_url"],
)


# -- Tools ---------------------------------------------------------------------


@crewai_tool("get_filter_context")
def get_filter_context(query: str) -> str:
    """Get available metadata fields and filter suggestions for a query.
    Call this first when the user mentions a company name, year, or document type."""
    return rag.get_filter_context(query)


@crewai_tool("search_documents")
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base and return relevant text chunks."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )


# -- Agent ---------------------------------------------------------------------

FINAL_ANSWER_INSTRUCTION = """You are an expert document analyst.
For complex questions — especially multi-company comparisons or multi-year analyses — break them into individual queries (one per company, one per year) and search for each separately before forming a combined answer.
If the query mentions a company name, year, or document type, call get_filter_context first.
You always search the knowledge base before answering and cite sources.
Bold all specific numbers, percentages, dates, and key financial figures using **value**.
Never wrap your response in code blocks or backticks.
If no documents are found, you say so honestly.
IMPORTANT: When you produce your final answer, start it with the exact marker "Final Answer:" on its own line.
After "Final Answer:", write only the answer text with no further tool calls or reasoning.
"""

agent = Agent(
    role="Document Assistant",
    goal="Answer user questions accurately using the document knowledge base, then output a final answer prefixed with 'Final Answer:'.",
    backstory=FINAL_ANSWER_INSTRUCTION,
    tools=[get_filter_context, search_documents],
    llm=llm,
    verbose=False,
)

# -- Helpers -------------------------------------------------------------------


_THINK_OPEN_TAG = "<" + "think" + ">"
_THINK_CLOSE_TAG = "</" + "think" + ">"
_FINAL_MARKER = re.compile(r"\bFinal\s*Answer\s*:", re.IGNORECASE)
_TOOL_TRACE_LINE = re.compile(
    r'^\s*"?((?:Thought|Action|Action Input|Observation|Tool|Tool Input|Tool Output))"?\s*:\s*',
    re.IGNORECASE,
)
_THINK_BLOCK = re.compile(
    _THINK_OPEN_TAG + ".*?" + _THINK_CLOSE_TAG,
    re.DOTALL | re.IGNORECASE,
)
_JSON_LINE = re.compile(r'^\s*(?:\{|\}|\[|\]|")')
_JSON_KEY_LINE = re.compile(r'^\s*"[^"]*"\s*:')
_CLOSING_CHARS = frozenset(["{", "}", "[", "]", "},", "],", "}{"])


def _strip_tool_trace(text: str) -> str:
    """Remove JSON tool-call/observation blocks emitted before the answer."""
    parts = re.split(r"\}(?!\s*[\{\"]\s)", text, maxsplit=1)
    if len(parts) > 1 and parts[1].strip():
        prose = parts[1].strip()
        if prose and prose[0].isalpha():
            return prose
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _CLOSING_CHARS:
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        if stripped.startswith("}") and stripped.endswith("{"):
            continue
        if re.match(r'^"[a-zA-Z_]+"\s*:', stripped):
            continue
        if stripped.startswith("{") or stripped.startswith("}"):
            after = re.sub(r"^[\s\{\}\[\]]+", "", stripped)
            if after and after[0].isalpha():
                lines[i] = after
                return "\n".join(lines[i:]).strip()
            continue
        if re.match(r"^[\{\}\[\]],?\s*$", stripped):
            continue
        return "\n".join(lines[i:]).strip()
    return text.strip()


def _is_only_tool_trace(text: str) -> bool:
    """Return True if text contains only tool-call JSON and no prose answer."""
    cleaned = _filter_lines(_strip_tool_trace(text))
    return not cleaned.strip()


def _extract_final_answer(raw: str) -> str:
    text = _THINK_BLOCK.sub("", raw)
    text = re.sub(
        _THINK_OPEN_TAG + ".*?" + _THINK_CLOSE_TAG,
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = _FINAL_MARKER.search(text)
    if match:
        text = text[match.end():]
        return _filter_lines(text)
    if _is_only_tool_trace(text):
        return ""
    text = _strip_tool_trace(text)
    return _filter_lines(text)


def _filter_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if _TOOL_TRACE_LINE.match(stripped):
            continue
        if _JSON_KEY_LINE.match(stripped):
            continue
        if _JSON_LINE.match(stripped):
            continue
        if stripped in _CLOSING_CHARS:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def last_user_message(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


# -- Public interface -----------------------------------------------------------
async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    task = Task(
        description=last_user_message(messages),
        expected_output=(
            "A detailed answer with source citations. "
            "Start your output with 'Final Answer:' followed by the answer."
        ),
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False, stream=True)
    streaming = await crew.kickoff_async()

    async for _ in streaming:
        pass

    raw = ""
    try:
        raw = streaming.result.raw
    except Exception:
        try:
            raw = streaming.get_full_text()
        except Exception:
            pass

    answer = _extract_final_answer(raw) if raw else ""

    if not answer:
        logger.warning("No extractable answer from CrewAI raw output (len=%d)", len(raw) if raw else 0)
        yield "I wasn't able to produce an answer for that query. Please try rephrasing."
        return

    if " " not in answer:
        yield answer
        return

    words = answer.split(" ")
    for i, word in enumerate(words):
        yield word if i == 0 else " " + word
        await asyncio.sleep(0.02)
