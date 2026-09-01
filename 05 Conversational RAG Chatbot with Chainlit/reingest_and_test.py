"""
Re-ingest all finance documents and test filtered retrieval.

Run from this directory:
    uv run reingest_and_test.py
"""

import os
from dotenv import load_dotenv

load_dotenv("../.env")

from ragwire import RAGWire, setup_logging

setup_logging(log_level="INFO")

MODEL_TIER = os.getenv("RAGWIRE_MODEL_TIER", "free")
rag = RAGWire("5config_openrouter_qdrant.yaml", model_tier=MODEL_TIER)
print(f"Using {rag.model_tier} model: {rag.config['llm']['model']}")

print("\n=== Ingesting documents ===")
stats = rag.ingest_directory("../data/finance_data")
print(f"Ingestion stats: {stats}")

print("\n=== Test 1: Filtered query (alphabet inc., 2024) ===")
results = rag.retrieve(
    query="what is revenue of Google in 2024?",
    filters={"company_name": "alphabet inc.", "fiscal_year": 2024},
)
print(f"Results: {len(results)}")
for r in results:
    m = r.metadata
    print(f"  company_name={m.get('company_name')!r}, fiscal_year={m.get('fiscal_year')!r}, file_name={m.get('file_name')!r}")

print("\n=== Test 2: Filtered query (apple inc., 2025) ===")
results = rag.retrieve(
    query="what is apple's revenue in 2025?",
    filters={"company_name": "apple inc.", "fiscal_year": 2025},
)
print(f"Results: {len(results)}")
for r in results:
    m = r.metadata
    print(f"  company_name={m.get('company_name')!r}, fiscal_year={m.get('fiscal_year')!r}, file_name={m.get('file_name')!r}")

print("\n=== Test 3: No filters ===")
results = rag.retrieve("what is revenue of Google in 2024?")
print(f"Results: {len(results)}")
for r in results:
    m = r.metadata
    print(f"  company_name={m.get('company_name')!r}, fiscal_year={m.get('fiscal_year')!r}, file_name={m.get('file_name')!r}")

print("\nDone.")
