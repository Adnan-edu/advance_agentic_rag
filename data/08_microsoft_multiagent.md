# 08_microsoft_multiagent.py — Explained

## What it does

This file implements a **parallel multi-agent RAG workflow** using the Microsoft Agent Framework. When a user asks a question, four specialist AI agents research it simultaneously from different angles, and a synthesizer merges their findings into one comprehensive answer.

---

## Architecture

```
User query
    │
    ▼
 [entry]  ──────────────────────────────────────────────────────┐
    │                                                            │
    ├──► specialist_financial  ──► collect_financial  ──►       │
    ├──► specialist_legal_risk ──► collect_legal_risk ──►  [aggregator]
    ├──► specialist_technical  ──► collect_technical  ──►       │
    └──► specialist_summary    ──► collect_summary    ──►       │
                                                                 │
                                                                 ▼
                                                          [synthesizer]
                                                                 │
                                                                 ▼
                                                          Final answer (streamed)
```

The workflow is an **acyclic DAG** (no loops). The Microsoft Agent Framework enforces this.

---

## Components

### 1. Constants and Configuration

```python
MODEL_ID = "ragwire-ms-supervisor"
GEMINI_MODEL = "models/gemini-2.5-flash"
```

- Uses **Google Gemini 2.5 Flash** as the LLM, accessed via its OpenAI-compatible endpoint.
- The API key is read from the `GOOGLE_API_KEY` environment variable.
- `SPECIALISTS` is a dict mapping each specialist's name to the topics it focuses on.

```python
SPECIALISTS = {
    "financial":  "revenue income profit margin financial statements cash flow",
    "legal_risk": "risk factors legal proceedings regulatory compliance liabilities",
    "technical":  "product technology research development innovation strategy",
    "summary":    "overview business strategy key highlights performance",
}
```

---

### 2. RAG Tools (`@tool`)

Two tools are available to every specialist agent:

| Tool | Purpose |
|---|---|
| `get_filter_context(query)` | Returns metadata fields (company, year, doc type) to use as filters. Call this first when the query mentions a specific company or year. |
| `search_documents(query, filters)` | Searches the vector store and returns the top-5 matching document chunks with their source filenames. |

These are registered with `@tool` so the Agent Framework can expose them to the LLM as callable functions.

---

### 3. `entry` Executor

```python
@executor(id="entry")
async def entry(message: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
```

- **Receives** the raw user query string.
- **Saves** it to shared workflow state (`QUERY_KEY`) so downstream executors can access it.
- **Initialises** an empty `specialist_outputs` dict in shared state.
- **Fans out** by sending an `AgentExecutorRequest` — the framework routes this to all four specialists in parallel because of the edges defined in `build_workflow`.

---

### 4. Specialist `AgentExecutor`s (×4)

```python
def make_specialist(name: str) -> AgentExecutor:
```

A factory that creates one `AgentExecutor` per specialist. Each specialist:

- Has a **focused system prompt** telling it which domain to cover.
- Has access to both RAG tools (`get_filter_context`, `search_documents`).
- Is instructed to bold all numbers/figures and cite source filenames.
- Runs its own LLM call independently and in parallel with the other three.

All four are created at module load time: `specialists = {name: make_specialist(name) for name in SPECIALISTS}`.

---

### 5. Collector Executors (×4)

```python
def make_collector(name: str) -> object:
    @executor(id=f"collect_{name}")
    async def collect(response: AgentExecutorResponse, ctx: ...) -> None:
```

A factory creating one collector per specialist. Each collector:

1. **Receives** the `AgentExecutorResponse` from its paired specialist.
2. **Extracts** the text via `response.agent_response.text`.
3. **Saves** it into the shared `specialist_outputs` dict keyed by specialist name.
4. **Forwards** an `AgentExecutorRequest` to the `aggregator`.

All four collectors write into the same shared dict — the aggregator uses the dict's length to know when all four are done.

---

### 6. `aggregator` Executor

```python
@executor(id="aggregator")
async def aggregator(_request: AgentExecutorRequest, ctx: ...) -> None:
```

This is a **manual fan-in gate**. It is called once by each collector (4 times total).

- **Checks** how many specialist outputs have been collected so far.
- **Returns early** (does nothing) for the first 3 calls — not all specialists are done yet.
- **On the 4th call**, all outputs are present. It:
  - Combines all four analyses into a single formatted string.
  - Sends one `AgentExecutorRequest` to the synthesizer containing the original query and all four analyses.

> **Why a manual aggregator instead of `add_fan_in_edges`?**
> The Microsoft Agent Framework's built-in `add_fan_in_edges` raises a `TypeCompatibilityError` when the source executors output `AgentExecutorRequest`. The manual aggregator is a robust workaround that achieves the same fan-in semantics.

---

### 7. `synthesizer` `AgentExecutor`

```python
synthesizer_exec = AgentExecutor(
    client.as_agent(name="Synthesizer", instructions=(...)),
    id="synthesizer",
)
```

- Receives the combined output from the aggregator.
- Its system prompt instructs it to merge all four analyses into one well-structured, cited answer.
- Bolds all key figures and formats references as `1. filename, p.XX`.

---

### 8. `build_workflow()`

```python
def build_workflow():
    builder = WorkflowBuilder(start_executor=entry)
    for name in SPECIALISTS:
        builder.add_edge(entry, specialists[name])          # fan-out
        builder.add_edge(specialists[name], collectors[name])
        builder.add_edge(collectors[name], aggregator)      # all 4 → aggregator
    builder.add_edge(aggregator, synthesizer_exec)          # aggregator → synthesizer
    return builder.build()
```

Defines the DAG by registering directed edges. The framework validates all edges at build time (type-checking input/output compatibility).

---

### 9. Public Interface

#### `stream(messages)` — `AsyncGenerator[str, None]`

```python
async def stream(messages: List[dict]) -> AsyncGenerator[str, None]:
    async for event in workflow.run(last_user(messages), stream=True):
        if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            if event.data.author_name == "Synthesizer" and event.data.text:
                yield event.data.text
```

- Extracts the last user message from the chat history.
- Runs the full workflow in **streaming mode**.
- Filters the event stream to only yield **token-by-token chunks from the Synthesizer**.
- All intermediate specialist outputs are silently processed and never streamed to the client.

#### `MODEL_ID` — `str`

Identifier string used by `routes.py` to label responses (`"ragwire-ms-supervisor"`).

---

## Data Flow Summary

| Step | Executor | Input | Output |
|---|---|---|---|
| 1 | `entry` | raw query string | `AgentExecutorRequest` (×4, fan-out) |
| 2 | `specialist_*` (×4, parallel) | `AgentExecutorRequest` | `AgentExecutorResponse` |
| 3 | `collect_*` (×4) | `AgentExecutorResponse` | saves text to state; sends `AgentExecutorRequest` to aggregator |
| 4 | `aggregator` (called 4×, fires once) | `AgentExecutorRequest` | `AgentExecutorRequest` (combined analyses) |
| 5 | `synthesizer` | `AgentExecutorRequest` | streamed `AgentResponseUpdate` tokens |

---

## Key Design Decisions

- **Parallel execution**: All four specialists run at the same time, cutting total latency to roughly the time of one LLM call instead of four sequential calls.
- **Shared state**: `WorkflowContext` is used as a lightweight key-value store to pass the original query and collected outputs between executors that don't directly connect.
- **Manual fan-in**: The `aggregator` replaces the framework's built-in `add_fan_in_edges` to avoid a type-compatibility validation bug.
- **Output filtering**: Only the Synthesizer's tokens are streamed to the caller; all specialist chatter stays internal.
