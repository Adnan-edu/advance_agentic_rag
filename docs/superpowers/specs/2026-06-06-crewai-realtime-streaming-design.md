# CrewAI Real-Time Streaming — Design

**Date**: 2026-06-06
**Target file**: `06 FastAPI RAG Backend/agents/04_crewai_agent.py`
**Addresses**: WORKFLOW.md problems #1 (streaming latency), #3 (reasoning leakage), #4 (no token-by-token streaming)

## Problem

The CrewAI agent's `stream()` collects every chunk from `crew.kickoff_async()` into a list, joins them, splits on `"Final Answer:"`, and yields the answer as one string. Users see nothing for 30–60 seconds, then the whole answer at once. When the marker is missing, raw output (including Thought/Action/Observation reasoning) is yielded verbatim.

CrewAI 1.6.1 (installed) streams `StreamChunk` objects with `content`, `chunk_type` (`TEXT` | `TOOL_CALL`), and `task_index`. The final answer arrives as TEXT chunks immediately after the literal marker `Final Answer:` in the token stream — so the answer can be streamed live by detecting the marker incrementally instead of after completion.

## Approach (chosen: marker state machine)

Rewrite `stream()` as a two-state incremental parser:

### State 1 — BUFFERING (before the marker)

- Append each TEXT chunk's content to a rolling buffer.
- After each append, search the buffer for `Final Answer:` (case-insensitive, optional surrounding whitespace/markdown bold like `**Final Answer:**`).
- The marker may be split across chunks (`"Final An"` + `"swer:"`), so the search must run on the accumulated buffer, not per-chunk.
- TOOL_CALL chunks are ignored (frontend shows Chainlit's default spinner; no status events).

### State 2 — STREAMING (after the marker)

- On first detection: yield everything in the buffer *after* the marker (lstripped), switch state.
- Every subsequent TEXT chunk is yielded immediately as received → real token-by-token streaming through FastAPI SSE → Chainlit `stream_token()`.
- No further marker searching.

### Fallback — stream ends with no marker

- If the agent never emits `Final Answer:` (some models answer directly), do not yield raw reasoning.
- Strip lines matching the existing `_REASONING_PATTERN` (`Thought:|Action:|Action Input:|Observation:` line prefixes) from the buffered text and yield the cleaned remainder.
- If nothing remains after cleaning, yield a friendly message: `"I wasn't able to produce an answer for that query. Please try rephrasing."` — never yield empty.

### Multi-task safety

- Track `chunk.task_index`; if it changes, reset to BUFFERING with a fresh buffer. The current crew has one task, so this is defensive only.

## What does NOT change

- `routes.py` SSE plumbing — already forwards each yielded string as a chunk; works unchanged.
- `app.py` Chainlit frontend — already streams tokens via `stream_token()`; works unchanged.
- Agent/tools/LLM configuration in `04_crewai_agent.py`.
- The public contract: `MODEL_ID: str`, `stream(messages) → AsyncGenerator[str, None]`.

## Component breakdown

| Unit | Responsibility |
|------|----------------|
| `FinalAnswerStreamFilter` (new class, pure/sync) | Incremental marker detection. `feed(text) → str` returns text safe to emit now; `flush() → str` returns fallback-cleaned text if marker never appeared. Holds buffer + state. No CrewAI imports — independently testable. |
| `stream()` (rewritten) | Async glue: builds Task/Crew as today, iterates chunks, routes TEXT content through the filter, yields non-empty emissions, calls `flush()` at end. |

`feed()` retains a small tail of the buffer (length of the longest marker variant) while BUFFERING so a split marker is still detected without unbounded rescanning.

## Error handling

- Exceptions from `kickoff_async()` propagate; `routes.py` already converts them to an `[Error: …]` SSE chunk (unchanged — broader error UX is WORKFLOW.md problem #5, out of scope).
- Empty stream → fallback message via `flush()`.

## Testing

Unit tests (`06 FastAPI RAG Backend/tests/test_crewai_stream_filter.py`) against `FinalAnswerStreamFilter` with synthetic chunk sequences:

1. Marker in one chunk → tail streamed, subsequent chunks pass through.
2. Marker split across two chunks (`"Final An"`, `"swer:"`).
3. `**Final Answer:**` bold variant and lowercase variant.
4. No marker, reasoning lines present → cleaned fallback on flush.
5. No marker, no reasoning → full text on flush.
6. Empty stream → friendly fallback message.
7. Reasoning text before marker is never emitted.

Manual E2E: run backend with `AGENT=04_crewai_agent`, ask a question via Chainlit, confirm tokens appear progressively and no Thought/Action text is shown.

## Out of scope

WORKFLOW.md problems #2 (model compatibility), #5 (error handling overhaul), #6 (context window), #7 (multi-agent) — future sub-projects.
