# Microsoft Agent Framework — Parallel Supervisor Multi-Agent Workflow

## Overview

This file implements a **fan-out / fan-in** multi-agent system using Microsoft Agent Framework's `WorkflowBuilder`. A user query is broadcast in parallel to four domain specialists, their outputs are collected into shared state, and once all four have reported, a **Synthesizer** agent produces the final consolidated answer.

---

## Architecture Diagram

```
                                USER QUERY
                                    |
                                    v
                          +---------+---------+
                          |  entry (executor) |
                          |                   |
                          |  1. Store query  |
                          |     in state     |
                          |  2. Initialize   |
                          |     outputs={}   |
                          |  3. Fan-out      |
                          |     message to   |
                          |     all 4 paths  |
                          +---------+---------+
                                    |
                 +------------------+------------------+------------------+
                 |                  |                  |                  |
                 v                  v                  v                  v
        +--------+------+  +-------+------+  +--------+------+  +--------+------+
        | specialist_   |  | specialist_  |  | specialist_   |  | specialist_  |
        |  financial    |  |  legal_risk  |  |  technical    |  |  summary     |
        |               |  |              |  |               |  |              |
        | LLM Agent     |  | LLM Agent    |  | LLM Agent     |  | LLM Agent    |
        | + RAG tools   |  | + RAG tools  |  | + RAG tools   |  | + RAG tools  |
        +--------+------+  +-------+------+  +--------+------+  +--------+------+
                 |                  |                  |                  |
                 v                  v                  v                  v
        +--------+------+  +-------+------+  +--------+------+  +--------+------+
        | collect_      |  | collect_     |  | collect_      |  | collect_     |
        |  financial    |  |  legal_risk  |  |  technical    |  |  summary     |
        |               |  |              |  |               |  |              |
        | 1. Read       |  | 1. Read      |  | 1. Read       |  | 1. Read      |
        |    outputs{}  |  |    outputs{} |  |    outputs{}  |  |    outputs{} |
        |    from state |  |    from state|  |    from state |  |    from state|
        | 2. Save this  |  | 2. Save this |  | 2. Save this  |  | 2. Save this |
        |    specialist |  |    specialist|  |    specialist |  |    specialist|
        |    answer     |  |    answer   |  |    answer     |  |    answer    |
        |    under key  |  |    under key |  |    under key  |  |    under key |
        | 3. Write dict |  | 3. Write dict|  | 3. Write dict |  | 3. Write dict|
        |    back to    |  |    back to   |  |    back to    |  |    back to   |
        |    state      |  |    state     |  |    state      |  |    state     |
        | 4. Forward    |  | 4. Forward   |  | 4. Forward    |  | 4. Forward   |
        |    query to   |  |    query to  |  |    query to   |  |    query to  |
        |    aggregator |  |    aggregator|  |    aggregator |  |    aggregator|
        +--------+------+  +-------+------+  +--------+------+  +--------+------+
                 |                  |                  |                  |
                 +------------------+------------------+------------------+
                                    |
                                    v
                          +---------+---------+
                          |   aggregator      |
                          |   (executor)      |
                          |                   |
                          | Check: are all   |
                          | 4 outputs in     |
                          | shared state?     |
                          |                   |
                          |  NO → return      |
                          |       (wait for   |
                          |        more)      |
                          |                   |
                          |  YES + not yet    |
                          |  fired → combine  |
                          |  all specialist   |
                          |  analyses, send   |
                          |  to Synthesizer   |
                          |  mark fired=True  |
                          +---------+---------+
                                    |
                                    v
                          +---------+---------+
                          |   synthesizer      |
                          |   (AgentExecutor) |
                          |                   |
                          | LLM Agent that    |
                          | merges all 4      |
                          | specialist reports|
                          | into one final    |
                          | structured answer |
                          +---------+---------+
                                    |
                                    v
                              FINAL ANSWER
                         (streamed to user)
```

---

## Step-by-Step Walkthrough

### Step 1 — Shared Client & Tools

```
client = OpenAIChatCompletionClient(...)
```

- A single `OpenAIChatCompletionClient` is created, pointed at OpenRouter.
- Two RAG tools (`get_filter_context`, `search_documents`) are defined and made available to specialist agents.

**Data format for tools:**

| Tool | Input | Output |
|---|---|---|
| `get_filter_context` | `query: str` | JSON string of available metadata filters |
| `search_documents` | `query: str, filters?: dict` | Concatenated document chunks with `[filename]` headers |

---

### Step 2 — Entry Executor

```
@executor(id="entry")
async def entry(message, ctx):
    ctx.set_state("query", message)          # Save original question
    ctx.set_state("specialist_outputs", {})   # Initialize empty outputs dict
    ctx.set_state("aggregator_fired", False)  # Guard against duplicate sends
    await ctx.send_message(                    # Fan-out to all specialists
        AgentExecutorRequest(
            messages=[Message(role="user", contents=[message])],
            should_respond=True
        )
    )
```

**What happens:**

| Action | State Key | Value | Purpose |
|---|---|---|---|
| Store query | `query` | `"What is Apple's earnings in 2024 and 2025?"` | Accessible by all downstream nodes |
| Init outputs | `specialist_outputs` | `{}` | Empty dict — collectors will populate this |
| Init guard | `aggregator_fired` | `False` | Prevents synthesizer from being called twice |
| Fan-out | — | `AgentExecutorRequest` | Broadcast to all 4 specialist edges |

Because the workflow graph has **four edges** from `entry` to each specialist, this single `send_message` fans out to all four simultaneously.

---

### Step 3 — Specialist AgentExecutors (Parallel)

```
SPECIALISTS = {
    "financial":  "revenue income profit margin financial statements cash flow",
    "legal_risk": "risk factors legal proceedings regulatory compliance liabilities",
    "technical":  "product technology research development innovation strategy",
    "summary":    "overview business strategy key highlights performance",
}
```

Each specialist is an `AgentExecutor` wrapping an LLM agent with:

- A **domain-specific system prompt** focusing on its area.
- **Both RAG tools** (`get_filter_context`, `search_documents`) so it can filter-then-search.
- A unique `id` like `specialist_financial`.

**Per-specialist execution flow:**

1. Receives the `AgentExecutorRequest` from entry.
2. The LLM decides whether to call `get_filter_context` first (if company/year/document type is mentioned).
3. Calls `search_documents` with appropriate filters to retrieve relevant chunks.
4. Produces a text answer citing sources and bolding key figures.

All four run **in parallel** — no sequential dependency between them.

---

### Step 4 — Collector Executors (Fan-in State Accumulation)

```
def make_collector(name):
    @executor(id=f"collect_{name}")
    async def collect(response, ctx):
        outputs = ctx.get_state("specialist_outputs") or {}  # Read shared dict
        outputs[name] = response.agent_response.text           # Save this specialist's answer
        ctx.set_state("specialist_outputs", outputs)           # Write back
        query = ctx.get_state("query") or ""                   # Original question
        await ctx.send_message(                                # Forward to aggregator
            AgentExecutorRequest(
                messages=[Message(role="user", contents=[query])],
                should_respond=True
            )
        )
    return collect
```

**Why a factory function?** Each collector needs its own unique `executor(id=)` so the workflow graph can route each specialist's output to the correct collector. A loop or dict comprehension can't create unique closures reliably, but `make_collector("financial")` etc. each produces a distinct `@executor`.

**Data flow in shared state** (example timeline):

```
After collect_financial runs:
  specialist_outputs = { "financial": "Revenue was **$394B**. Source: Apple_10k_2024.pdf" }

After collect_legal_risk runs:
  specialist_outputs = {
    "financial": "Revenue was **$394B**. Source: Apple_10k_2024.pdf",
    "legal_risk": "Apple faces regulatory risk in EU. Source: Apple_10k_2025.pdf"
  }

After collect_summary runs (assuming technical finishes last):
  specialist_outputs = {
    "financial": "...",
    "legal_risk": "...",
    "technical": "...",
    "summary": "..."
  }
```

Each collector forwards `query` (not the specialist's output — that's already in shared state) to the aggregator via `send_message`.

---

### Step 5 — Aggregator Executor (Gatekeeper)

```
@executor(id="aggregator")
async def aggregator(_request, ctx):
    outputs = ctx.get_state("specialist_outputs") or {}
    if len(outputs) < 4:          # Not all specialists done yet
        return                    # Wait — do nothing
    if ctx.get_state("aggregator_fired"):  # Already sent once
        return                    # Prevent duplicate synthesizer calls
    ctx.set_state("aggregator_fired", True)

    query = ctx.get_state("query")
    combined = "\n\n".join(
        f"## {name.upper()} ANALYSIS\n{text}"
        for name, text in outputs.items()
    )
    await ctx.send_message(
        AgentExecutorRequest(
            messages=[Message(role="user",
                contents=[f"Query: {query}\n\nSpecialist Analyses:\n{combined}"])],
            should_respond=True
        )
    )
```

**The aggregator is called once per collector** (4 times total). It acts as a gatekeeper:

| Call # | `len(outputs)` | `aggregator_fired` | Action |
|---|---|---|---|
| 1st | 1 | `False` | `return` (too early) |
| 2nd | 2 | `False` | `return` (too early) |
| 3rd | 3 | `False` | `return` (too early) |
| 4th | 4 | `False` | Combine all outputs, send to Synthesizer, set `aggregator_fired = True` |

The `aggregator_fired` guard is critical: even though `len(outputs) == 4` on the 4th call, if there were a 5th `send_message` from any path, the guard prevents sending a duplicate request to the Synthesizer.

**Message format sent to Synthesizer:**

```
Query: What is Apple's earnings in 2024 and 2025?

Specialist Analyses:
## FINANCIAL ANALYSIS
Revenue was **$394B** in 2024...

## LEGAL_RISK ANALYSIS
Apple faces regulatory risk in EU...

## TECHNICAL ANALYSIS
Apple invested **$30B** in R&D...

## SUMMARY ANALYSIS
Apple's overall performance...
```

---

### Step 6 — Synthesizer AgentExecutor

```
synthesizer_exec = AgentExecutor(
    client.as_agent(
        name="Synthesizer",
        instructions="Synthesize the specialist analyses into one comprehensive...",
    ),
    id="synthesizer",
)
```

The Synthesizer:

1. Receives the combined prompt with all 4 specialist analyses + the original query.
2. Produces a single unified answer with citations, bolded figures, and no code blocks.
3. Its output is streamed to the user via the `stream()` function.

---

### Step 7 — Workflow Graph Construction

```
builder = WorkflowBuilder(start_executor=entry)
for name in SPECIALISTS:
    builder.add_edge(entry, specialists[name])           # entry → specialist
    builder.add_edge(specialists[name], collectors[name]) # specialist → collector
    builder.add_edge(collectors[name], aggregator)        # collector → aggregator
builder.add_edge(aggregator, synthesizer_exec)           # aggregator → synthesizer
workflow = builder.build()
```

**Edge map (DAG):**

```
entry ──────────► specialist_financial  ──► collect_financial ──► aggregator
entry ──────────► specialist_legal_risk  ──► collect_legal_risk  ──► aggregator
entry ──────────► specialist_technical   ──► collect_technical   ──► aggregator
entry ──────────► specialist_summary     ──► collect_summary     ──► aggregator
aggregator ──────► synthesizer
```

Key constraint: **Microsoft Agent Framework uses acyclic DAGs** — no loop-back cycles allowed. This is why the aggregator pattern (count arrivals in shared state) is used instead of a loop.

---

### Step 8 — Streaming Output

```
async def stream(messages):
    async for event in workflow.run(last_user_message(messages), stream=True):
        if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            if event.data.author_name == "Synthesizer" and event.data.text:
                yield event.data.text
```

- `workflow.run()` streams `WorkflowEvent` objects.
- `event.type == "output"` filters for agent output events.
- `event.data.author_name == "Synthesizer"` ensures only the final answer is yielded — all specialist intermediate reasoning is suppressed.

---

## Shared State Summary

| Key | Type | Set By | Purpose |
|---|---|---|---|
| `query` | `str` | `entry` | Original user question, read by collectors and aggregator |
| `specialist_outputs` | `dict[str, str]` | `collect_*` (each adds its key) | Accumulates all specialist answers; read by aggregator |
| `aggregator_fired` | `bool` | `entry` (init `False`), `aggregator` (set `True`) | Prevents duplicate Synthesizer invocations |

---

## Key Design Decisions

1. **Why collectors instead of direct specialist → aggregator edges?**
   The `AgentExecutor` emits `AgentExecutorResponse`, but the aggregator needs access to **shared state** (`specialist_outputs` dict). Collectors serve as lightweight adapter nodes that: (a) extract text from the agent response, (b) store it in shared state under the specialist's name, and (c) forward the query onward.

2. **Why `aggregator_fired` guard?**
   Each of the 4 collectors sends a message to the aggregator. On the 4th arrival, `len(outputs) == 4` triggers the Synthesizer. But without the guard, a hypothetical 5th message (if any path retriggers) would send a duplicate. The flag ensures exactly-once execution.

3. **Why parallel fan-out instead of sequential?**
   All four specialists operate on the same query but focus on different domains. Running them in parallel reduces total latency from `4 × T` to roughly `1 × T` (where `T` is the slowest specialist).

4. **Why DAG instead of loop?**
   Microsoft Agent Framework enforces acyclic graphs. The "supervisor" pattern (loop back to refine) is approximated by having the Synthesizer receive all specialist outputs at once and produce a final consolidated answer in a single pass.