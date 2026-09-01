# Repository-wide OpenRouter model-tier migration

## Purpose

Apply the model-selection pattern established in `02 RAGWire Setup and First Retrieval` to the other relevant notebooks, YAML configurations, applications, scripts, and agent implementations in this repository.

This is a migration plan, not permanent agent behavior, so it belongs in `PLANS.md` rather than `AGENTS.md`.

## Target outcome

Every OpenRouter-backed workflow must select its model through one explicit tier:

- `free` -> `nvidia/nemotron-3-super-120b-a12b:free`
- `paid` -> `nex-agi/nex-n2-mini`

Notebooks should select the tier with a constructor argument:

```python
MODEL_TIER = "free"  # "free" or "paid"
rag = RAGWire("config.yaml", model_tier=MODEL_TIER)
```

Long-running applications and CLI scripts should select it through an environment variable and pass the value to the same constructor:

```python
MODEL_TIER = os.getenv("RAGWIRE_MODEL_TIER", "free")
rag = RAGWire(CONFIG_PATH, model_tier=MODEL_TIER)
```

Use `OPENROUTER_API_KEY` for authentication. Do not copy it into `OPENAI_API_KEY`, and do not set `OPENAI_BASE_URL` as an application-wide side effect.

## Non-negotiable rules

1. Never fall back from `free` to `paid` automatically. A hidden fallback can create unexpected charges. Paid usage must be an explicit user choice.
2. Do not hard-code a model again after `RAGWire` has selected one.
   - LangChain/LangGraph code must use `rag.llm`.
   - Other SDKs must use `rag.config["llm"]["model"]` as their model ID.
3. Keep embeddings independent from the LLM tier. Changing free/paid chat models does not require a new Qdrant collection because vector compatibility is determined by the embedding model.
4. Do not set `force_recreate: true` or delete Qdrant data merely because the LLM changed.
5. Enable `metadata.fail_on_extraction_error: true` for metadata-driven RAG lessons. A failed LLM extraction must not silently store chunks that cannot be filtered.
6. Free NVIDIA endpoints may log prompts. Use public or non-sensitive documents only and preserve the privacy warning in notebooks and documentation.
7. Validate tool calling separately for every agent framework. A model working through LangChain does not prove that CrewAI, AutoGen, or Microsoft Agent Framework will encode its tools correctly.
8. Preserve intentional provider demonstrations. Lesson 03 is a provider-comparison lesson; do not replace its OpenAI, Gemini, Groq, or Ollama examples wholesale.
9. Preserve unrelated user changes in the dirty worktree. Inspect `git status` and the relevant diffs before every edit.

## Canonical YAML shape

Use this structure in OpenRouter-backed RAGWire configurations:

```yaml
llm:
  provider: "openrouter"
  model_tier: "free"
  model_options:
    free: "nvidia/nemotron-3-super-120b-a12b:free"
    paid: "nex-agi/nex-n2-mini"
  api_key: "${OPENROUTER_API_KEY}"
  base_url: "https://openrouter.ai/api/v1"
  max_retries: 6

metadata:
  config_file: "finance_metadata.yaml"
  fail_on_extraction_error: true
```

Use the domain-specific metadata file already associated with each lesson. Do not blindly replace `finance_metadata.yaml` with another schema.

## Repository inventory and intended treatment

| Area | Files | Required treatment |
|---|---|---|
| Canonical package | `ragwire/core/pipeline.py`, `ragwire/__init__.py` | Keep `model_tier`, `model_options`, `rag.model_tier`, and public `rag.llm` as the single contract. Add automated tests before migrating consumers. |
| Lesson 02 baseline | `02 RAGWire Setup and First Retrieval/config.yaml`, `RAGWire Setup and First Retrieval.ipynb` | Treat as the reference implementation. Verify only; do not duplicate model constructors. |
| Lesson 03 provider cookbook | `03 RAGWire in Practice - Providers, Components, and Cookbooks/` | Preserve the existing provider-specific configs. Add a clearly separate OpenRouter tier example/config if desired. Replace the final hard-coded `ChatOpenAI(model="gpt-4.1-mini")` agent with the LLM belonging to the selected RAGWire instance. |
| Lesson 04 health RAG | `config_openrouter_qdrant.yaml`, `Personal Gym Supplements RAG.ipynb`, `reingest_missing.py`, supporting docs | Migrate YAML to the canonical tier shape; remove OpenAI environment aliases; add `MODEL_TIER`; use `rag.llm` for LangChain agents; pass the tier to re-ingestion. Keep the health metadata schema. |
| Lesson 04 shadow package | `04 Personal Gym Supplements RAG/ragwire/` | This stale package can shadow the canonical root package when the notebook runs from lesson 04. Verify import resolution, then remove or quarantine this copy instead of maintaining two divergent RAGWire implementations. Do not delete it without first checking whether any uncommitted lesson-specific logic exists. |
| Lesson 05 Chainlit RAG | `5config_openrouter_qdrant.yaml`, `RAGWire Ingest Documents.ipynb`, `app.py`, `reingest_and_test.py` | Migrate YAML; pass `RAGWIRE_MODEL_TIER`; remove `OPENAI_*` aliases and duplicate `ChatOpenAI`; use `rag.llm` in `create_agent`. |
| Lesson 06 FastAPI backend | `config/6config_openrouter_qdrant.yaml`, `tools.py`, `agents/01_*.py` through `agents/08_*.py` | Make `tools.py` construct the one tier-selected `rag`. LangChain/LangGraph agents use `rag.llm`; CrewAI, AutoGen, and Microsoft adapters read the selected ID from `rag.config["llm"]["model"]`. Remove competing model defaults such as `OPENROUTER_MODEL_ID` and `CREWAI_MODEL_ID`, unless retained only as documented expert overrides. |
| Lesson 07 frontend | `07 Chainlit Chat Frontend/` and `WORKFLOW.md` | The frontend should not select an LLM because the FastAPI backend owns inference. Update documentation and environment examples only. Do not add a second model selector to the UI. |
| Provider backups | `02/.../config_bkp_detailed.yaml`, lesson 03 provider configs | Leave intentional Ollama/OpenAI/Gemini/Groq examples intact. Label them clearly rather than silently converting them. |

## Execution phases

### Phase 0: protect the worktree and verify import ownership

- Record `git status --short` and inspect diffs for every target file.
- From the repository root and from each lesson directory, run:

```bash
python -c "import ragwire; print(ragwire.__file__)"
```

- The import must resolve to the canonical root package or the editable `.venv` installation.
- Investigate the lesson 04 package copy before removal. Compare it with root `ragwire/`; preserve any unique behavior deliberately, not by keeping an accidental fork.
- Do not touch PDFs, databases, Qdrant collections, generated reports, or unrelated application work during this migration.

#### Local Qdrant guardrail

When the intended database is the repository's local Docker Qdrant, do not reuse a
`${QDRANT_URL}` cloud configuration merely because it exists in `.env`:

1. Inspect `start_qdrant.sh` and `scripts/check_qdrant.py`; local mode is
   `http://localhost:6333` and can be verified with `uv run scripts/check_qdrant.py`.
2. Check the local collection name and vector schema before creating a new lesson
   config. Its embedding provider/model and vector dimension must match the existing
   collection. A mismatch can create a collection or require paid embedding calls.
3. Keep `force_recreate: false`. Do not initialize a config that could create, delete,
   or re-index a collection without first reporting the collection name and operation.
4. `start_qdrant.sh` migrates legacy container-local storage into the persistent
   `qdrant_storage` volume and retains a recovery container. Do not delete that
   recovery container until collection health and retrieval are verified.

### Phase 1: lock down the canonical model-tier contract

Add focused tests for the root package:

- omitted argument uses `llm.model_tier` from YAML;
- `model_tier="free"` selects Nemotron;
- `model_tier="paid"` selects Nex-N2-Mini;
- invalid values fail with a useful error;
- passing `model_tier` to a legacy config without `model_options` fails clearly;
- legacy configs without a tier continue to use `llm.model`;
- `rag.llm` is the same model instance used by `MetadataExtractor`;
- `max_retries` and the OpenRouter base URL reach `ChatOpenAI`;
- `fail_on_extraction_error: true` marks the file failed and prevents vector upsert.

Do not begin repository-wide consumer edits until these deterministic tests pass.

### Phase 2: migrate OpenRouter YAML configurations

Update these configs to `provider: openrouter` plus `model_tier` and `model_options`:

- `04 Personal Gym Supplements RAG/config_openrouter_qdrant.yaml`
- `05 Conversational RAG Chatbot with Chainlit/5config_openrouter_qdrant.yaml`
- `06 FastAPI RAG Backend/config/6config_openrouter_qdrant.yaml`

For each config:

- retain its current embedding provider and vector dimension;
- retain its existing Qdrant URL, API key placeholder, and collection name;
- retain its domain metadata YAML;
- add `max_retries: 6`;
- add `metadata.fail_on_extraction_error: true`;
- do not rename or recreate the collection unless the embedding model also changes.

Every lesson notebook that uses OpenRouter must have a separate, clearly named
OpenRouter config (normally `config_openrouter_qdrant.yaml`; preserve an established
lesson prefix such as `5config_openrouter_qdrant.yaml`). Never rewrite a provider-
comparison config to add OpenRouter. Lesson 03 must remain a provider cookbook: its
OpenRouter path uses
`03 RAGWire in Practice - Providers, Components, and Cookbooks/config_openrouter_qdrant.yaml`;
the intentional `config_openai*.yaml`, Gemini, Groq, and Ollama examples remain intact.

### Phase 3: migrate notebooks safely

Target notebooks:

- `03 RAGWire in Practice - Providers, Components, and Cookbooks/RAGWire in Practice.ipynb`
- `04 Personal Gym Supplements RAG/Personal Gym Supplements RAG.ipynb`
- `05 Conversational RAG Chatbot with Chainlit/RAGWire Ingest Documents.ipynb`

For each OpenRouter workflow:

1. Add one early selection cell:

```python
MODEL_TIER = "free"  # change to "paid" explicitly when desired
rag = RAGWire(CONFIG_PATH, model_tier=MODEL_TIER)
print(f"Using {rag.model_tier} model: {rag.config['llm']['model']}")
```

2. Replace manually constructed `ChatOpenAI(...)` objects with `rag.llm`.
3. Remove unused `ChatOpenAI` and `os` imports when they are no longer needed.
4. Remove `OPENAI_API_KEY = OPENROUTER_API_KEY` and `OPENAI_BASE_URL` mutation cells.
5. Update prose to document both tiers, pricing risk, and the NVIDIA free-endpoint privacy warning.
6. Clear cached outputs and reset execution counts after editing. Old outputs must not claim that a stale provider/model is active.
7. Preserve the Python 3.13 `.venv` kernel metadata.
8. Compile every code cell after JSON parsing; do not rely only on the notebook opening successfully.

Lesson 03 exception: preserve cells whose explicit purpose is to demonstrate OpenAI, Gemini, Groq, or Ollama. Only apply tier selection to a distinct OpenRouter path.

### Phase 4: migrate Python applications and scripts

#### Lesson 04

- `reingest_missing.py`:
  - add `--model-tier {free,paid}` with default from `RAGWIRE_MODEL_TIER`;
  - pass it to `RAGWire`;
  - remove OpenAI environment aliases;
  - print the selected tier and model before any destructive delete/re-ingest work.
- `check_missing_metadata.py` and `delete_missing_metadata_records.py` do not invoke an LLM. Do not add model-selection code to them.
- Keep deletion dry-run-first and require explicit user authority before deleting Qdrant points.

#### Lesson 05

- `app.py`:
  - initialize `rag` with `RAGWIRE_MODEL_TIER`;
  - delete the separate `ChatOpenAI` construction;
  - pass `model=rag.llm` to `create_agent`;
  - remove `OPENAI_*` environment aliases.
- `reingest_and_test.py`:
  - read `RAGWIRE_MODEL_TIER` or add `--model-tier`;
  - pass it to `RAGWire`;
  - remove OpenAI aliases.
- The ingestion notebook follows the notebook rules from Phase 3.

#### Lesson 06

- `tools.py` becomes the process-level source of truth:

```python
MODEL_TIER = os.getenv("RAGWIRE_MODEL_TIER", "free")
rag = RAGWire(CONFIG_PATH, model_tier=MODEL_TIER)
SELECTED_MODEL_ID = rag.config["llm"]["model"]
```

- LangChain/LangGraph agents (`01`, `02`, `03`) should use `rag.llm` rather than create another `ChatOpenAI`.
- CrewAI agents (`04`, `05`) should build their SDK-specific wrapper from `rag.config["llm"]["model"]`, preserving any required `openrouter/` prefix.
- AutoGen (`06`) and Microsoft Agent Framework (`07`, `08`) should use the same selected model ID while retaining framework-specific capability declarations.
- Remove model-specific defaults that can disagree with the RAG pipeline. Prefer only `RAGWIRE_MODEL_TIER` as the ordinary operator control.
- Do not assume free-tier tool compatibility. Run a real synthetic tool-call smoke test for every framework. If a framework adapter cannot use Nemotron correctly, fail startup with a precise message telling the operator to choose `RAGWIRE_MODEL_TIER=paid`; do not silently switch tiers.

### Phase 5: update documentation and environment examples

Update relevant sections in:

- `README.md`
- `04 Personal Gym Supplements RAG/METADATA_README.md`
- `07 Chainlit Chat Frontend/WORKFLOW.md`
- any lesson-specific run instructions discovered during implementation

Document:

```dotenv
OPENROUTER_API_KEY=...
RAGWIRE_MODEL_TIER=free
```

Explain that `paid` requires OpenRouter credit and creates billable requests. Do not commit `.env` or expose key values.

Reconcile existing claims that Nemotron cannot call tools with actual per-framework test results. Do not delete a warning merely because LangChain passed; CrewAI or another adapter may still fail.

## Validation matrix

### Static validation

```bash
uv sync --check
python -m compileall -q ragwire \
  "04 Personal Gym Supplements RAG" \
  "05 Conversational RAG Chatbot with Chainlit" \
  "06 FastAPI RAG Backend"
git diff --check
```

Parse every notebook as JSON and compile every Python code cell. Ignore notebook magics deliberately rather than treating them as Python syntax.

Search for drift after migration:

```bash
rg -n --glob '!uv.lock' \
  'z-ai/glm-5.2:free|OPENAI_API_KEY.*OPENROUTER|OPENAI_BASE_URL|OPENROUTER_MODEL_ID|CREWAI_MODEL_ID|ChatOpenAI\(' \
  .
```

Every remaining match must have a documented reason, such as an intentional provider lesson or an SDK adapter that cannot reuse `rag.llm`.

### Deterministic tests

- Run the root unit tests for tier resolution and metadata fail-fast behavior.
- Add application tests that mock external LLM calls and assert every agent receives the selected model ID.
- Verify paid selection without sending a paid request unless the user explicitly authorizes cost.

### Live smoke tests

Use only synthetic or public text:

1. Free-tier structured metadata extraction.
2. Free-tier tool call through each supported agent framework.
3. RAGWire initialization against the intended Qdrant instance.
4. One retrieval against existing data, without ingestion mutation.
5. Paid-tier live calls only with explicit approval.

Record model ID, adapter, result, and failure type. Distinguish authentication errors, platform rate limits, upstream capacity limits, schema incompatibility, and tool-call incompatibility.

### Data-quality validation

- Audit each collection for records missing required domain metadata.
- If repair is necessary, report affected file hashes and point counts first.
- Use dry-run deletion/re-ingestion tooling before any destructive action.
- Never hide extraction failures by ingesting chunks with only system metadata in configurations that enable `fail_on_extraction_error`.

## Definition of done

- All OpenRouter-backed RAGWire configs expose the same free/paid tier mapping.
- All relevant notebooks choose a tier once and reuse `rag.llm`.
- Application and agent processes use `RAGWIRE_MODEL_TIER` as their normal selection control.
- No OpenRouter workflow aliases credentials into `OPENAI_API_KEY`.
- No stale hard-coded GLM or duplicate Nemotron constructors remain outside explicitly documented examples.
- Intentional provider-comparison lessons still work as provider comparisons.
- The lesson 04 shadow package no longer masks the canonical root package.
- Metadata failures cannot silently create unfilterable records in migrated configs.
- Static checks, deterministic tests, notebook validation, and approved live smoke tests pass.
- Documentation states free-endpoint privacy limitations and paid-tier cost behavior.

## Explicitly out of scope

- Replacing every non-OpenRouter provider example.
- Changing embedding models or Qdrant vector dimensions.
- Deleting or recreating collections without a separately reviewed data-migration reason.
- Automatically spending money when a free provider is unavailable.
- Editing generated reports, PDFs, databases, or unrelated frontend behavior.
