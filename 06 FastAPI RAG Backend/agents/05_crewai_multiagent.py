from typing import AsyncGenerator, List, Optional

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool as crewai_tool
from crewai.types.streaming import StreamChunkType
from tools import SELECTED_MODEL_ID, rag


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "ragwire-crewai-multi"

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = LLM(
    model=f"openrouter/{SELECTED_MODEL_ID}",
    api_key=rag.config["llm"]["api_key"],
    base_url=rag.config["llm"]["base_url"],
    timeout=120,
    max_retries=3,
)

# ── Tools (Researcher only) ────────────────────────────────────────────────────

@crewai_tool("get_filter_context")
def get_filter_context(query: str) -> str:
    """Get available metadata fields and filter suggestions for a query.
    Call this first when the user mentions a company name, year, or document type."""
    return rag.get_filter_context(query)


@crewai_tool("search_documents")
# def search_documents(query: str) -> str:
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base and return relevant text chunks.
    For multi-company or multi-year questions, call once per company/year."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )

# ── Helpers ────────────────────────────────────────────────────────────────────

def _last_user(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


# ── Agents ─────────────────────────────────────────────────────────────────────
researcher = Agent(
    role="Document Researcher",
    goal="Retrieve all relevant raw information from the knowledge base for the given query.",
    backstory="""
        You are a meticulous document researcher. You always search the knowledge base before drawing any conclusions.
        If the query mentions a company name, year, or document type, call get_filter_context first to get available metadata filters.
        For complex queries, you break them into sub-queries and call search_documents separately for each.
        You return all retrieved facts, numbers, and source filenames in full — you never summarize or interpret.
        """,
    tools=[get_filter_context, search_documents],
    llm=llm,
    verbose=False,
)

analyst = Agent(
    role="Research Analyst",
    goal="Extract key insights, structure findings, and identify gaps from the raw research.",
    backstory="""
        You are a sharp analyst. You receive raw document excerpts from the researcher and organize them into clear, structured findings.
        You identify the most important facts, highlight specific numbers and dates, and flag any missing information.
        You never fabricate data — if something isn't in the research, you say so.
        """,
    llm=llm,
    verbose=False,
)

writer = Agent(
    role="Technical Writer",
    goal="Compose a clear, well-structured final answer with source citations.",
    backstory="""
        You are an expert technical writer. You take structured analyst findings and craft a polished, readable final answer.
        You always cite source documents.
        Bold all specific numbers, percentages, dates, and key figures using **value**.
        Never wrap your response in code blocks or backticks.
        If the research found no relevant information, you say so honestly.
        """,
    llm=llm,
    verbose=False,
)



def build_crew(query: str) -> Crew:
    research_task = Task(
        description=f"Search the knowledge base for all information relevant to: {query}",
        expected_output="Raw document excerpts with source filenames. No interpretation.",
        agent=researcher,
    )
    analysis_task = Task(
        description="Analyze the research findings and structure the key insights.",
        expected_output="Structured bullet-point findings with specific facts, numbers, and source references.",
        agent=analyst,
        context=[research_task],
    )
    writing_task = Task(
        description=f"Write the final answer to this question: {query}",
        expected_output="A clear, well-structured answer with source citations and bolded key figures.",
        agent=writer,
        context=[analysis_task],
    )
    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=False,
        stream=True,
    )

# ── Public interface ───────────────────────────────────────────────────────────

async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    query = _last_user(messages)
    crew = build_crew(query)
    streaming = await crew.kickoff_async()
    async for chunk in streaming:
        if (
            chunk.content
            and chunk.chunk_type == StreamChunkType.TEXT
            and chunk.agent_role == writer.role
        ):
            yield chunk.content
