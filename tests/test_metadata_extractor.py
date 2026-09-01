"""Tests for MetadataExtractor content sampling and escalation."""

from typing import Optional

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from ragwire.metadata.extractor import MetadataExtractor


class SampleSchema(BaseModel):
    title: Optional[str] = None
    year: Optional[int] = None


class FakeLLM:
    """Fake chat model: records each prompt and replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def with_structured_output(self, schema):
        def _invoke(prompt_value):
            self.prompts.append(prompt_value.to_string())
            return self.responses[len(self.prompts) - 1]

        return RunnableLambda(_invoke)


def make_long_text():
    """Document longer than the prefix window, with a heading beyond it."""
    body = "word " * 1000  # ~5000 chars of filler
    return (
        "# Creatine Study\n\nAuthors: Smith et al.\n\n"
        + body
        + "\n## Resistance Training Protocol\n\nmore text\n"
        + body
        + "\n## Conclusions\n\nfinal text\n"
    )


# --- _build_content_sample ---


def test_short_text_returned_unchanged():
    text = "# Title\n\nshort document"
    assert MetadataExtractor._build_content_sample(text) == text


def test_long_text_keeps_prefix_and_appends_heading_outline():
    text = make_long_text()
    sample = MetadataExtractor._build_content_sample(text)

    assert sample.startswith("# Creatine Study")
    assert "## Document Outline" in sample
    # Heading that lies beyond the 3k prefix must appear via the outline
    assert "## Resistance Training Protocol" in sample
    assert "## Conclusions" in sample
    # Sample must stay far smaller than the full document
    assert len(sample) < len(text)


def test_outline_is_capped():
    headings = "\n".join(f"# Heading number {i} with some extra words" for i in range(200))
    text = "intro " * 600 + "\n" + headings
    sample = MetadataExtractor._build_content_sample(text)
    outline = sample.split("## Document Outline\n", 1)[1]
    assert len(outline) <= 1000


def test_long_text_without_headings_falls_back_to_prefix_only():
    text = "plain text " * 1000
    sample = MetadataExtractor._build_content_sample(text)
    assert sample == text[:3000]
    assert "## Document Outline" not in sample


# --- extract() escalation ---


def test_extract_uses_sample_not_full_text():
    llm = FakeLLM([SampleSchema(title="creatine study", year=2024)])
    extractor = MetadataExtractor(llm, schema_model=SampleSchema)
    text = make_long_text()

    metadata = extractor.extract(text)

    assert metadata == {"title": "creatine study", "year": 2024}
    assert len(llm.prompts) == 1
    # The far tail of the document must not be in the prompt
    assert "final text" not in llm.prompts[0]
    # But the outline carried deep headings to the LLM
    assert "## Resistance Training Protocol" in llm.prompts[0]


def test_extract_escalates_once_when_all_fields_null():
    llm = FakeLLM([
        SampleSchema(title=None, year=None),
        SampleSchema(title="creatine study", year=2024),
    ])
    extractor = MetadataExtractor(llm, schema_model=SampleSchema)
    text = make_long_text()

    metadata = extractor.extract(text)

    assert len(llm.prompts) == 2
    # Second pass sees a larger window: deep content present this time
    assert "Resistance Training Protocol" in llm.prompts[1]
    assert len(llm.prompts[1]) > len(llm.prompts[0])
    assert metadata == {"title": "creatine study", "year": 2024}


def test_extract_does_not_escalate_when_fields_found():
    llm = FakeLLM([SampleSchema(title="found", year=2024)])
    extractor = MetadataExtractor(llm, schema_model=SampleSchema)

    extractor.extract(make_long_text())

    assert len(llm.prompts) == 1


def test_extract_escalates_on_missing_required_field():
    llm = FakeLLM([
        SampleSchema(title=None, year=2024),  # year found, but title is required
        SampleSchema(title="creatine study", year=2024),
    ])
    extractor = MetadataExtractor(
        llm, schema_model=SampleSchema, required_fields=["title"]
    )

    metadata = extractor.extract(make_long_text())

    assert len(llm.prompts) == 2
    assert metadata["title"] == "creatine study"


def test_extract_with_required_fields_ignores_other_nulls():
    llm = FakeLLM([SampleSchema(title="found", year=None)])
    extractor = MetadataExtractor(
        llm, schema_model=SampleSchema, required_fields=["title"]
    )

    extractor.extract(make_long_text())

    assert len(llm.prompts) == 1  # year=None is fine, title was required and found


def test_escalation_keeps_first_pass_values_when_second_returns_null():
    llm = FakeLLM([
        SampleSchema(title=None, year=2024),
        SampleSchema(title="creatine study", year=None),  # second pass loses year
    ])
    extractor = MetadataExtractor(
        llm, schema_model=SampleSchema, required_fields=["title"]
    )

    metadata = extractor.extract(make_long_text())

    assert metadata == {"title": "creatine study", "year": 2024}


def test_no_escalation_when_document_fits_in_sample():
    llm = FakeLLM([
        SampleSchema(title=None, year=None),
        SampleSchema(title=None, year=None),
    ])
    extractor = MetadataExtractor(llm, schema_model=SampleSchema)

    extractor.extract("# Short\n\ntiny doc")

    assert len(llm.prompts) == 1  # a bigger window has nothing more to show


def test_escalation_window_is_capped():
    llm = FakeLLM([
        SampleSchema(title=None, year=None),
        SampleSchema(title="creatine study", year=2024),
    ])
    extractor = MetadataExtractor(llm, schema_model=SampleSchema)
    # Document much longer than the escalation window, with a marker past it
    text = "intro " * 3000 + "BEYOND_ESCALATION_MARKER"
    assert len(text) > MetadataExtractor.ESCALATION_CHARS

    extractor.extract(text)

    assert len(llm.prompts) == 2
    # Retry sees more than the first-pass sample, but never past the cap
    assert "BEYOND_ESCALATION_MARKER" not in llm.prompts[1]


def test_escalation_retry_value_overrides_first_pass():
    llm = FakeLLM([
        SampleSchema(title="partial title", year=None),  # year missing → retry
        SampleSchema(title="full corrected title", year=2024),
    ])
    extractor = MetadataExtractor(
        llm, schema_model=SampleSchema, required_fields=["year"]
    )

    metadata = extractor.extract(make_long_text())

    # When both passes return a value, the retry's value wins
    assert metadata == {"title": "full corrected title", "year": 2024}


def test_escalation_runs_at_most_once():
    llm = FakeLLM([
        SampleSchema(title=None, year=None),
        SampleSchema(title=None, year=None),  # retry also finds nothing
    ])
    extractor = MetadataExtractor(llm, schema_model=SampleSchema)

    metadata = extractor.extract(make_long_text())

    assert len(llm.prompts) == 2  # exactly one retry, never a third call
    assert metadata == {"title": None, "year": None}


# --- from_yaml required flag ---


def test_from_yaml_reads_required_flags(tmp_path):
    yaml_file = tmp_path / "meta.yaml"
    yaml_file.write_text(
        "fields:\n"
        "  - name: title\n"
        "    description: Document title\n"
        "    required: true\n"
        "  - name: year\n"
        "    description: Publication year\n"
        "    type: integer\n"
    )
    llm = FakeLLM([])
    extractor = MetadataExtractor.from_yaml(llm, str(yaml_file))

    assert extractor.required_fields == ["title"]
    assert extractor.fields == ["title", "year"]
