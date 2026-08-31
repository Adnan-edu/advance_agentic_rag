"""
RAG tools - shared across all agent implementations.

Usage:
    from tools import rag, get_filter_context, search_documents
"""

import os
from typing import Optional

from langchain.tools import tool

from ragwire import RAGWire

# ── RAGWire ───────────────────────────────────────────────────────────────────

CONFIG_PATH = "model_tier_config/config/config_openrouter_qdrant.yaml"
MODEL_TIER = os.getenv("RAGWIRE_MODEL_TIER", "free")

rag = RAGWire(CONFIG_PATH, model_tier=MODEL_TIER)
SELECTED_MODEL_ID = rag.config["llm"]["model"]

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_filter_context(query: str) -> str:
    """Get available metadata fields and filter suggestions for a query.
    Call this first when the user mentions a company name, year, or document type."""
    return rag.get_filter_context(query)



@tool
def search_documents(query: str, filters: Optional[dict] = None) -> str:
    """Search the document knowledge base and return relevant text chunks.
    For multi-company comparisons, call once per company. For multi-year analyses, call once per year.
    Each call should be a focused query targeting one company, one year, or one specific aspect."""
    results = rag.retrieve(query, top_k=5, filters=filters)
    if not results:
        return "No relevant documents found."
    chunks = [
        f"[{doc.metadata.get('file_name', 'unknown')}]\n{doc.page_content}"
        for doc in results
    ]
    return "\n\n---\n\n".join(chunks)
