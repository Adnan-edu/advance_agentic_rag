"""Deterministic checks for lesson 06 model-tier wiring.

These tests replace RAGWire during ``tools.py`` import, so they never contact an
LLM or Qdrant instance.
"""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest import TestCase, mock

import yaml


BACKEND_DIR = Path(__file__).resolve().parents[1]


class FakeRAGWire:
    instances = []

    def __init__(self, config_path: str, model_tier: str):
        models = {
            "free": "nvidia/nemotron-3-super-120b-a12b:free",
            "paid": "nex-agi/nex-n2-mini",
        }
        self.config_path = config_path
        self.model_tier = model_tier
        self.config = {
            "llm": {
                "model": models[model_tier],
                "api_key": "test-key",
                "base_url": "https://openrouter.ai/api/v1",
            }
        }
        self.llm = object()
        self.instances.append(self)


def load_tools(model_tier: str | None):
    fake_ragwire = ModuleType("ragwire")
    fake_ragwire.RAGWire = FakeRAGWire

    fake_langchain = ModuleType("langchain")
    fake_langchain.__path__ = []
    fake_langchain_tools = ModuleType("langchain.tools")
    fake_langchain_tools.tool = lambda func: func

    spec = importlib.util.spec_from_file_location(
        f"lesson06_tools_{model_tier or 'default'}", BACKEND_DIR / "tools.py"
    )
    module = importlib.util.module_from_spec(spec)
    with (
        mock.patch.dict(
            sys.modules,
            {
                "ragwire": fake_ragwire,
                "langchain": fake_langchain,
                "langchain.tools": fake_langchain_tools,
            },
        ),
        mock.patch.dict(
            os.environ,
            {} if model_tier is None else {"RAGWIRE_MODEL_TIER": model_tier},
            clear=True,
        ),
    ):
        spec.loader.exec_module(module)
    return module


class ModelTierWiringTests(TestCase):
    def setUp(self):
        FakeRAGWire.instances.clear()

    def test_tools_passes_free_tier_to_ragwire(self):
        tools = load_tools("free")
        self.assertEqual(
            FakeRAGWire.instances[0].config_path,
            "model_tier_config/config/config_openrouter_qdrant.yaml",
        )
        self.assertEqual(FakeRAGWire.instances[0].model_tier, "free")
        self.assertEqual(
            tools.SELECTED_MODEL_ID,
            "nvidia/nemotron-3-super-120b-a12b:free",
        )

    def test_tools_passes_paid_tier_without_calling_provider(self):
        tools = load_tools("paid")
        self.assertEqual(FakeRAGWire.instances[0].model_tier, "paid")
        self.assertEqual(tools.SELECTED_MODEL_ID, "nex-agi/nex-n2-mini")

    def test_config_has_canonical_tiers_and_preserves_qdrant_settings(self):
        config = yaml.safe_load(
            (
                BACKEND_DIR
                / "model_tier_config/config/config_openrouter_qdrant.yaml"
            ).read_text()
        )
        self.assertEqual(config["llm"]["provider"], "openrouter")
        self.assertEqual(config["llm"]["model_tier"], "free")
        self.assertEqual(
            config["llm"]["model_options"],
            {
                "free": "nvidia/nemotron-3-super-120b-a12b:free",
                "paid": "nex-agi/nex-n2-mini",
            },
        )
        self.assertEqual(config["llm"]["max_retries"], 6)
        self.assertTrue(config["metadata"]["fail_on_extraction_error"])
        self.assertEqual(config["embeddings"]["model_name"], "BAAI/bge-m3")
        self.assertEqual(config["vectorstore"]["url"], "http://localhost:6333")
        self.assertEqual(
            config["vectorstore"]["collection_name"], "finance-rag-qdrant"
        )
        self.assertFalse(config["vectorstore"]["force_recreate"])
        self.assertEqual(
            config["metadata"]["config_file"],
            "model_tier_config/metadata/finance_metadata.yaml",
        )

    def test_every_agent_uses_the_process_selected_model(self):
        expected = {
            "01_langchain_agent.py": "model=rag.llm",
            "02_langgraph_self_correcting_agent.py": "rag.llm.ainvoke",
            "03_langgraph_supervisor_agent.py": "rag.llm.ainvoke",
            "04_crewai_agent.py": 'model=f"openrouter/{SELECTED_MODEL_ID}"',
            "05_crewai_multiagent.py": 'model=f"openrouter/{SELECTED_MODEL_ID}"',
            "06_autogen_agent.py": "model=SELECTED_MODEL_ID",
            "07_microsoft_agent.py": "model=SELECTED_MODEL_ID",
            "08_microsoft_multiagent.py": "model=SELECTED_MODEL_ID",
        }
        stale_controls = ("OPENROUTER_MODEL_ID", "CREWAI_MODEL_ID", "ChatOpenAI(")

        for filename, required_wiring in expected.items():
            source = (BACKEND_DIR / "agents" / filename).read_text()
            compact = "".join(source.split())
            with self.subTest(agent=filename):
                self.assertIn("".join(required_wiring.split()), compact)
                for stale_control in stale_controls:
                    self.assertNotIn(stale_control, source)

    def test_tools_defaults_to_free_tier(self):
        tools = load_tools(None)
        self.assertEqual(FakeRAGWire.instances[0].model_tier, "free")
        self.assertEqual(
            tools.SELECTED_MODEL_ID,
            "nvidia/nemotron-3-super-120b-a12b:free",
        )
