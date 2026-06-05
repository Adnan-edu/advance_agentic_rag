# CrewAI Real-Time Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream the CrewAI agent's final answer token-by-token as it is generated, instead of buffering the entire agent run, while never leaking Thought/Action/Observation reasoning text.

**Architecture:** A pure, synchronous `FinalAnswerStreamFilter` class incrementally detects the `Final Answer:` marker in the chunk stream (even split across chunks or bolded) and switches from buffering to pass-through streaming. The rewritten `stream()` in `04_crewai_agent.py` routes CrewAI TEXT chunks through this filter and yields emissions immediately. `routes.py` (SSE) and `app.py` (Chainlit) already forward per-chunk tokens and need no changes.

**Tech Stack:** Python 3, CrewAI 1.6.1 (`crew.kickoff_async()` streaming, `StreamChunk`/`StreamChunkType`), FastAPI SSE, pytest.

**Spec:** `docs/superpowers/specs/2026-06-06-crewai-realtime-streaming-design.md`

**Environment:** Activate the venv first for every command:
```bash
source "/Users/mdashikadnan/Documents/adnanedu/python/udemy/AI/ragwire/ragwire/.venv/bin/activate"
```
All paths below are relative to the project root `/Users/mdashikadnan/Documents/adnanedu/python/udemy/AI/ragwire/ragwire`. Note the directory `06 FastAPI RAG Backend` contains spaces — always quote paths.

**Important import constraint:** The agents directory files start with digits (`04_crewai_agent.py`) and the backend is run with `cwd` = `06 FastAPI RAG Backend`. The filter module will live in a new file `06 FastAPI RAG Backend/agents/stream_filter.py` (importable name). Tests import it by adding the backend dir to `sys.path` via a `conftest.py`.

---

### Task 1: Test scaffolding (conftest)

**Files:**
- Create: `06 FastAPI RAG Backend/tests/__init__.py` (empty)
- Create: `06 FastAPI RAG Backend/tests/conftest.py`

- [ ] **Step 1: Create the tests package and conftest**

Create empty file `06 FastAPI RAG Backend/tests/__init__.py`.

Create `06 FastAPI RAG Backend/tests/conftest.py`:

```python
"""Make the backend directory importable so tests can import agents.stream_filter."""
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
```

- [ ] **Step 2: Verify pytest collects (no tests yet, exit code 5 is fine)**

Run: `cd "06 FastAPI RAG Backend" && python -m pytest tests/ -v`
Expected: "no tests ran" (exit code 5) — no import errors.

- [ ] **Step 3: Commit**

```bash
git add "06 FastAPI RAG Backend/tests/__init__.py" "06 FastAPI RAG Backend/tests/conftest.py"
git commit -m "test: add pytest scaffolding for backend tests"
```

---

### Task 2: FinalAnswerStreamFilter — marker detection in a single chunk

**Files:**
- Create: `06 FastAPI RAG Backend/agents/stream_filter.py`
- Create: `06 FastAPI RAG Backend/tests/test_stream_filter.py`

- [ ] **Step 1: Write the failing tests**

Create `06 FastAPI RAG Backend/tests/test_stream_filter.py`:

```python
from agents.stream_filter import FinalAnswerStreamFilter


def test_marker_in_single_chunk_emits_tail():
    f = FinalAnswerStreamFilter()
    out = f.feed("Thought: I should answer now.\nFinal Answer: Revenue grew 12%.")
    assert out == "Revenue grew 12%."


def test_text_before_marker_is_not_emitted():
    f = FinalAnswerStreamFilter()
    assert f.feed("Thought: I need to search the knowledge base.\n") == ""
    assert f.feed("Action: search_documents\n") == ""


def test_chunks_after_marker_pass_through_verbatim():
    f = FinalAnswerStreamFilter()
    f.feed("Final Answer: The company")
    assert f.feed(" reported **$5B**") == " reported **$5B**"
    assert f.feed(" in revenue.") == " in revenue."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "06 FastAPI RAG Backend" && python -m pytest tests/test_stream_filter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.stream_filter'`

- [ ] **Step 3: Write minimal implementation**

Create `06 FastAPI RAG Backend/agents/stream_filter.py`:

```python
"""Incremental filter that detects CrewAI's "Final Answer:" marker in a token
stream and passes through only the final answer, never the reasoning text.

Pure/synchronous — no CrewAI imports — so it is independently testable.
"""

import re

# Matches "Final Answer:" with optional markdown bold and flexible spacing,
# case-insensitively (e.g. "Final Answer:", "**Final Answer:**", "final answer :").
_MARKER = re.compile(r"\*{0,2}final answer\*{0,2}\s*:\*{0,2}", re.IGNORECASE)


class FinalAnswerStreamFilter:
    """Two-state incremental parser.

    BUFFERING: accumulate text, search for the marker on each feed.
    STREAMING: pass every chunk through verbatim.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._streaming = False

    def feed(self, text: str) -> str:
        """Feed one chunk; return whatever is safe to emit now ("" if nothing)."""
        if self._streaming:
            return text
        self._buffer += text
        match = _MARKER.search(self._buffer)
        if match:
            self._streaming = True
            tail = self._buffer[match.end():]
            self._buffer = ""
            return tail.lstrip()
        return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "06 FastAPI RAG Backend" && python -m pytest tests/test_stream_filter.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add "06 FastAPI RAG Backend/agents/stream_filter.py" "06 FastAPI RAG Backend/tests/test_stream_filter.py"
git commit -m "feat: add FinalAnswerStreamFilter with single-chunk marker detection"
```

---

### Task 3: Marker split across chunks + formatting variants

**Files:**
- Modify: `06 FastAPI RAG Backend/tests/test_stream_filter.py`
- Modify: `06 FastAPI RAG Backend/agents/stream_filter.py` (only if tests fail)

- [ ] **Step 1: Add the failing/edge tests**

Append to `06 FastAPI RAG Backend/tests/test_stream_filter.py`:

```python
def test_marker_split_across_two_chunks():
    f = FinalAnswerStreamFilter()
    assert f.feed("Thought: done.\nFinal An") == ""
    out = f.feed("swer: The answer is 42.")
    assert out == "The answer is 42."


def test_marker_split_across_three_chunks():
    f = FinalAnswerStreamFilter()
    assert f.feed("Fin") == ""
    assert f.feed("al Answ") == ""
    assert f.feed("er: Done.") == "Done."


def test_bold_marker_variant():
    f = FinalAnswerStreamFilter()
    out = f.feed("Thought: ok\n**Final Answer:** The total is **$9M**.")
    assert out == "The total is **$9M**."


def test_lowercase_marker_variant():
    f = FinalAnswerStreamFilter()
    out = f.feed("final answer: yes.")
    assert out == "yes."
```

- [ ] **Step 2: Run tests**

Run: `cd "06 FastAPI RAG Backend" && python -m pytest tests/test_stream_filter.py -v`
Expected: all pass (the buffer-accumulation design from Task 2 already handles splits; bold/lowercase covered by the regex). If any fail, fix `_MARKER` or `feed()` minimally until green.

- [ ] **Step 3: Commit**

```bash
git add "06 FastAPI RAG Backend/tests/test_stream_filter.py" "06 FastAPI RAG Backend/agents/stream_filter.py"
git commit -m "test: cover split-marker and formatting variants in stream filter"
```

---

### Task 4: flush() fallback — no marker, reasoning cleanup, empty stream

**Files:**
- Modify: `06 FastAPI RAG Backend/tests/test_stream_filter.py`
- Modify: `06 FastAPI RAG Backend/agents/stream_filter.py`

- [ ] **Step 1: Write the failing tests**

Append to `06 FastAPI RAG Backend/tests/test_stream_filter.py`:

```python
def test_flush_no_marker_strips_reasoning_lines():
    f = FinalAnswerStreamFilter()
    f.feed("Thought: I should search.\n")
    f.feed("Action: search_documents\n")
    f.feed("Action Input: {\"query\": \"revenue\"}\n")
    f.feed("Observation: [doc.pdf] Revenue was $5B.\n")
    f.feed("The revenue was **$5B** in 2024.\n")
    assert f.flush() == "The revenue was **$5B** in 2024."


def test_flush_no_marker_no_reasoning_returns_full_text():
    f = FinalAnswerStreamFilter()
    f.feed("The answer is 42.")
    assert f.flush() == "The answer is 42."


def test_flush_empty_stream_returns_fallback_message():
    f = FinalAnswerStreamFilter()
    assert f.flush() == FinalAnswerStreamFilter.FALLBACK_MESSAGE


def test_flush_only_reasoning_returns_fallback_message():
    f = FinalAnswerStreamFilter()
    f.feed("Thought: hmm\nAction: search_documents\n")
    assert f.flush() == FinalAnswerStreamFilter.FALLBACK_MESSAGE


def test_flush_after_streaming_returns_empty():
    f = FinalAnswerStreamFilter()
    f.feed("Final Answer: Done.")
    assert f.flush() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "06 FastAPI RAG Backend" && python -m pytest tests/test_stream_filter.py -v`
Expected: new tests FAIL — `AttributeError: ... no attribute 'flush'`

- [ ] **Step 3: Implement flush()**

Add to `FinalAnswerStreamFilter` in `06 FastAPI RAG Backend/agents/stream_filter.py` (module-level constant goes near `_MARKER`):

```python
_REASONING_LINE = re.compile(r"^\s*(Thought|Action|Action Input|Observation)\s*:", re.IGNORECASE)
```

And the class additions:

```python
    FALLBACK_MESSAGE = (
        "I wasn't able to produce an answer for that query. Please try rephrasing."
    )

    def flush(self) -> str:
        """Call once when the stream ends.

        If the marker was found, everything was already emitted — return "".
        Otherwise strip reasoning lines from the buffered text and return the
        remainder, or a friendly fallback if nothing useful is left.
        """
        if self._streaming:
            return ""
        cleaned = "\n".join(
            line for line in self._buffer.splitlines()
            if not _REASONING_LINE.match(line)
        ).strip()
        self._buffer = ""
        return cleaned if cleaned else self.FALLBACK_MESSAGE
```

- [ ] **Step 4: Run all filter tests**

Run: `cd "06 FastAPI RAG Backend" && python -m pytest tests/test_stream_filter.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add "06 FastAPI RAG Backend/tests/test_stream_filter.py" "06 FastAPI RAG Backend/agents/stream_filter.py"
git commit -m "feat: add flush() fallback with reasoning cleanup to stream filter"
```

---

### Task 5: Multi-task reset (defensive)

**Files:**
- Modify: `06 FastAPI RAG Backend/tests/test_stream_filter.py`
- Modify: `06 FastAPI RAG Backend/agents/stream_filter.py`

- [ ] **Step 1: Write the failing test**

Append to `06 FastAPI RAG Backend/tests/test_stream_filter.py`:

```python
def test_reset_returns_to_buffering_with_fresh_buffer():
    f = FinalAnswerStreamFilter()
    f.feed("Final Answer: first task answer.")
    f.reset()
    # After reset, reasoning from the next task must be buffered again, not passed through.
    assert f.feed("Thought: starting second task\n") == ""
    assert f.feed("Final Answer: second answer.") == "second answer."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "06 FastAPI RAG Backend" && python -m pytest tests/test_stream_filter.py::test_reset_returns_to_buffering_with_fresh_buffer -v`
Expected: FAIL — `AttributeError: ... no attribute 'reset'`

- [ ] **Step 3: Implement reset()**

Add to the class in `06 FastAPI RAG Backend/agents/stream_filter.py`:

```python
    def reset(self) -> None:
        """Return to BUFFERING with a fresh buffer (new task started)."""
        self._buffer = ""
        self._streaming = False
```

- [ ] **Step 4: Run all filter tests**

Run: `cd "06 FastAPI RAG Backend" && python -m pytest tests/test_stream_filter.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add "06 FastAPI RAG Backend/tests/test_stream_filter.py" "06 FastAPI RAG Backend/agents/stream_filter.py"
git commit -m "feat: add reset() for multi-task safety in stream filter"
```

---

### Task 6: Rewire stream() in 04_crewai_agent.py

**Files:**
- Modify: `06 FastAPI RAG Backend/agents/04_crewai_agent.py:93-112` (the `stream()` function), plus imports at top

- [ ] **Step 1: Update imports**

In `06 FastAPI RAG Backend/agents/04_crewai_agent.py`, the current top-of-file has an unused `asyncio` import and a module-level `_REASONING_PATTERN` regex that is superseded by the filter. Replace:

```python
import asyncio
import re
from typing import AsyncGenerator, List, Optional

from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool as crewai_tool
from crewai.types.streaming import StreamChunkType
import os
from dotenv import load_dotenv
from tools import rag

load_dotenv()

_REASONING_PATTERN = re.compile(r'(Thought|Action|Observation|Action Input)\s*:', re.MULTILINE)
```

with:

```python
from typing import AsyncGenerator, List, Optional

from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool as crewai_tool
from crewai.types.streaming import StreamChunkType
import os
from dotenv import load_dotenv
from tools import rag

from agents.stream_filter import FinalAnswerStreamFilter

load_dotenv()
```

- [ ] **Step 2: Replace the stream() function**

Replace the entire existing `stream()` (lines 93-112, the `async def stream(...)` through end of file) with:

```python
async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    task = Task(
        description=last_user_message(messages),
        expected_output="A detailed answer with source citations.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False, stream=True)
    streaming = await crew.kickoff_async()

    answer_filter = FinalAnswerStreamFilter()
    current_task_index = None
    async for chunk in streaming:
        if chunk.task_index != current_task_index:
            if current_task_index is not None:
                answer_filter.reset()
            current_task_index = chunk.task_index
        if chunk.content and chunk.chunk_type == StreamChunkType.TEXT:
            emission = answer_filter.feed(chunk.content)
            if emission:
                yield emission

    remainder = answer_filter.flush()
    if remainder:
        yield remainder
```

- [ ] **Step 3: Verify the module still imports cleanly**

Run: `cd "06 FastAPI RAG Backend" && python -c "import importlib; m = importlib.import_module('agents.04_crewai_agent'); print(m.MODEL_ID)"`
Expected: prints `ragwire-crewai` (requires `OPENROUTER_API_KEY` in `.env`, which already exists).

- [ ] **Step 4: Run the full test suite**

Run: `cd "06 FastAPI RAG Backend" && python -m pytest tests/ -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add "06 FastAPI RAG Backend/agents/04_crewai_agent.py"
git commit -m "feat: stream CrewAI final answer in real time via FinalAnswerStreamFilter"
```

---

### Task 7: Manual E2E verification (with google/gemini-2.5-flash)

**Files:** none modified (verification only; `.env` may be edited by the user)

- [ ] **Step 1: Start the backend with the CrewAI agent**

```bash
cd "06 FastAPI RAG Backend"
source ../.venv/bin/activate
AGENT=04_crewai_agent python main.py
```

To test the tutorial's model, also set `CREWAI_MODEL_ID=google/gemini-2.5-flash` (env var or `.env`). The design is model-agnostic — the `Final Answer:` marker comes from CrewAI's prompt format, not the model.

- [ ] **Step 2: Smoke-test the SSE endpoint directly**

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What documents do you have access to?"}]}'
```

Expected: multiple `data: {...}` chunks arriving **progressively** (not one big chunk at the end), no `Thought:`/`Action:`/`Observation:` text in any `delta.content`.

- [ ] **Step 3: Verify in Chainlit**

```bash
cd "07 Chainlit Chat Frontend"
source ../.venv/bin/activate
chainlit run app.py
```

Open http://localhost:8000, log in, ask a question. Expected: spinner while the agent reasons/calls tools, then the answer appears token-by-token. Repeat with `CREWAI_MODEL_ID=google/gemini-2.5-flash` and confirm identical behavior.

- [ ] **Step 4: Update WORKFLOW.md**

In `07 Chainlit Chat Frontend/WORKFLOW.md`, update the "Streaming Implementation" code sample (section 3) to match the new `stream()`, and mark problems #1, #3, #4 in "Current Problems and Limitations" as resolved with a one-line note pointing to `agents/stream_filter.py`.

- [ ] **Step 5: Commit**

```bash
git add "07 Chainlit Chat Frontend/WORKFLOW.md"
git commit -m "docs: mark streaming problems resolved in WORKFLOW.md"
```
