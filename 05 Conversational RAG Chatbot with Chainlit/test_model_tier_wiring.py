"""Deterministic tests for Lesson 05 model-tier and Chainlit wiring."""

import asyncio
import importlib.util
import os
import tempfile
import tomllib
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.documents import Document

from ragwire import RAGWire


class Lesson05AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.captured = {}
        cls.selected_llm = object()

        class FakeRAGWire:
            def __init__(self, config_path, model_tier=None):
                cls.captured["config_path"] = config_path
                cls.captured["model_tier"] = model_tier
                cls.captured["initialization_cwd"] = str(Path.cwd())
                self.llm = cls.selected_llm

            def get_filter_context(self, query):
                return ""

            def retrieve(self, query, filters=None):
                return []

            def ingest_directory(self, directory):
                return {}

        cls.app_path = Path(__file__).with_name("app.py").resolve()
        spec = importlib.util.spec_from_file_location(
            "lesson05_app_model_tier_test", cls.app_path
        )
        cls.app = importlib.util.module_from_spec(spec)
        original_cwd = Path.cwd()

        with (
            patch.dict(os.environ, {"RAGWIRE_MODEL_TIER": "paid"}),
            patch("ragwire.RAGWire", FakeRAGWire),
        ):
            spec.loader.exec_module(cls.app)

        cls.cwd_was_restored = Path.cwd() == original_cwd

    def test_app_passes_selected_tier_and_reuses_rag_llm(self):
        self.assertEqual(
            Path(self.captured["config_path"]),
            self.app_path.with_name("5config_openrouter_qdrant.yaml"),
        )
        self.assertEqual(self.captured["model_tier"], "paid")
        self.assertEqual(
            Path(self.captured["initialization_cwd"]), self.app_path.parent
        )
        self.assertTrue(self.cwd_was_restored)

        with patch.object(
            self.app, "create_agent", return_value=object()
        ) as create_agent:
            self.app.build_agent()

        kwargs = create_agent.call_args.kwargs
        self.assertIs(kwargs["model"], self.selected_llm)
        self.assertEqual(len(kwargs["middleware"]), 1)
        retry = kwargs["middleware"][0]
        self.assertEqual(retry.max_retries, 2)
        self.assertEqual(retry.on_failure, "error")
        self.assertIs(retry.retry_on, self.app.is_transient_model_error)

    def test_transient_error_classification_and_safe_messages(self):
        overloaded = ValueError(
            {"message": "Upstream error: Service temporarily overloaded", "code": 502}
        )
        unauthorized = RuntimeError("401 invalid API key: secret-value")

        self.assertTrue(self.app.is_transient_model_error(overloaded))
        self.assertFalse(self.app.is_transient_model_error(unauthorized))
        self.assertIn("temporarily overloaded", self.app.user_facing_model_error(overloaded))
        safe_message = self.app.user_facing_model_error(unauthorized)
        self.assertNotIn("secret-value", safe_message)

    def test_search_results_are_bounded_and_source_labeled(self):
        documents = [
            Document(
                page_content=(f"evidence-{index} " * 500),
                metadata={
                    "file_name": f"report-{index}.pdf",
                    "company_name": "example inc.",
                    "doc_type": "10-k",
                    "fiscal_year": 2025,
                },
            )
            for index in range(6)
        ]

        output = self.app.format_search_results(documents)

        self.assertEqual(output.count("[Result "), 5)
        self.assertIn("Source: report-0.pdf", output)
        self.assertNotIn("report-5.pdf", output)
        self.assertIn("…", output)

    def test_ingestion_summary_reports_partial_failure(self):
        summary = self.app.format_ingestion_summary(
            {
                "processed": 1,
                "skipped": 2,
                "failed": 1,
                "chunks_created": 12,
                "errors": [{"file": "/tmp/failed.pdf", "error": "provider error"}],
            }
        )

        self.assertIn("1 processed, 2 skipped, 1 failed", summary)
        self.assertIn("failed.pdf", summary)
        self.assertNotIn("provider error", summary)

    def test_upload_validation_sanitizes_and_rejects_unsafe_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pdf"
            source.write_text("public test document")
            element = SimpleNamespace(path=str(source), name="../../report.PDF")

            uploads = self.app.validate_uploads([element])
            self.assertEqual(uploads, [(source, "report.PDF")])

            with self.assertRaisesRegex(ValueError, "Unsupported file type"):
                self.app.validate_uploads(
                    [SimpleNamespace(path=str(source), name="payload.exe")]
                )
            with self.assertRaisesRegex(ValueError, "Duplicate upload filename"):
                self.app.validate_uploads([element, element])
            with (
                patch.object(self.app, "MAX_UPLOAD_BYTES", 1),
                self.assertRaisesRegex(ValueError, "50 MB upload limit"),
            ):
                self.app.validate_uploads([element])

    def test_model_failure_is_shown_without_escaping_callback(self):
        agent = SimpleNamespace(
            ainvoke=AsyncMock(
                side_effect=ValueError("Upstream service temporarily overloaded")
            )
        )
        messages = []

        class FakeMessage:
            def __init__(self, content):
                self.content = content
                messages.append(self)

            async def send(self):
                return None

            async def update(self):
                return None

        def get_session_value(key):
            return agent if key == "agent" else "thread-test"

        with (
            patch.object(self.app.cl.user_session, "get", side_effect=get_session_value),
            patch.object(self.app.cl, "Message", FakeMessage),
            patch.object(self.app.logger, "exception") as log_exception,
        ):
            asyncio.run(
                self.app.on_message(SimpleNamespace(elements=[], content="question"))
            )

        self.assertIn("temporarily overloaded", messages[-1].content)
        log_exception.assert_called_once()

    def test_upload_ingestion_uses_async_wrappers_and_honest_summary(self):
        calls = []
        messages = []
        stats = {
            "processed": 0,
            "skipped": 0,
            "failed": 1,
            "chunks_created": 0,
            "errors": [{"file": "/tmp/report.pdf", "error": "test failure"}],
        }

        class FakeMessage:
            def __init__(self, content):
                self.content = content
                messages.append(self)

            async def send(self):
                return None

            async def update(self):
                return None

        def fake_make_async(function):
            async def run(*args, **kwargs):
                calls.append(function)
                return function(*args, **kwargs)

            return run

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "report.pdf"
            source.write_text("public test document")
            incoming = SimpleNamespace(path=str(source), name="report.pdf")
            message = SimpleNamespace(elements=[incoming], content="")

            with (
                patch.object(self.app.cl.user_session, "get", return_value=None),
                patch.object(self.app.cl, "Message", FakeMessage),
                patch.object(self.app.cl, "make_async", side_effect=fake_make_async),
                patch.object(
                    self.app.rag, "ingest_directory", Mock(return_value=stats)
                ) as ingest,
            ):
                asyncio.run(self.app.on_message(message))

        self.assertIs(calls[0], self.app.copy_uploads)
        self.assertIs(calls[1], ingest)
        self.assertIn("1 failed", messages[-1].content)
        self.assertIn("report.pdf", messages[-1].content)

    def test_chainlit_upload_configuration_matches_server_limits(self):
        config_path = self.app_path.parent / ".chainlit" / "config.toml"
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)

        uploads = config["features"]["spontaneous_file_upload"]
        self.assertEqual(uploads["max_files"], self.app.MAX_UPLOAD_FILES)
        self.assertEqual(uploads["max_size_mb"], 50)
        self.assertNotIn("*/*", uploads["accept"])


class Lesson05ConfigTest(unittest.TestCase):
    def test_config_resolves_free_and_paid_without_external_calls(self):
        config_path = Path(__file__).with_name("5config_openrouter_qdrant.yaml")
        expected_models = {
            "free": "nvidia/nemotron-3-super-120b-a12b:free",
            "paid": "nex-agi/nex-n2-mini",
        }
        component_initializers = (
            "_initialize_logging",
            "_initialize_loader",
            "_initialize_splitter",
            "_initialize_embeddings",
            "_initialize_llm",
            "_initialize_vectorstore",
            "_initialize_retriever",
        )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-openrouter-key"}):
            for tier, expected_model in expected_models.items():
                with self.subTest(tier=tier), ExitStack() as stack:
                    for initializer in component_initializers:
                        stack.enter_context(patch.object(RAGWire, initializer))
                    rag = RAGWire(str(config_path), model_tier=tier)

                self.assertEqual(rag.model_tier, tier)
                self.assertEqual(rag.config["llm"]["model"], expected_model)
                self.assertEqual(
                    rag.config["vectorstore"]["url"], "http://localhost:6333"
                )


if __name__ == "__main__":
    unittest.main()
