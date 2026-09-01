# Missing Metadata Processing Pipeline

End-to-end workflow for detecting, deleting, and re-ingesting Qdrant records that have null metadata fields (title, authors, publication_year, research_focus).

---

## Architecture

```
check_missing_metadata.py          Step 1: Scan Qdrant and report null fields
         |
         v
reingest_missing.py               Step 2: Auto-detect, delete stale points, re-ingest (one command)
   or manually:
     delete_missing_metadata_records.py   Step 2a: Remove stale points from Qdrant
     reingest_missing.py --skip-delete    Step 2b: Re-ingest files through RAGWire pipeline
         |
         v
check_missing_metadata.py          Step 3: Verify no null fields remain
```

All three scripts read connection details from `config_openrouter_qdrant.yaml` and use the same environment variable resolution (`${QDRANT_URL}`, `${QDRANT_API_KEY}`, `${OPENROUTER_API_KEY}`).

`reingest_missing.py` auto-detects files with missing metadata by querying Qdrant — no hardcoded file list needed. It resolves file names to paths on disk, deletes stale points, and re-ingests in a single command.

---

## Pipeline Stages

### Stage 1 — Detect

```bash
python check_missing_metadata.py
```

| What it does | Scans every point in the Qdrant collection and checks whether any metadata field among `title`, `authors`, `publication_year`, `research_focus` is null. Groups results by file and writes a detailed Markdown report. |
|---|---|
| Config read | `config_openrouter_qdrant.yaml` (Qdrant URL, API key, collection name) |
| Output | `missing_metadata_report.md` (overwritten each run) |
| Exit codes | 0 on success; prints total records and missing count to stdout |

The report contains:
- Collection name, total records, records with missing metadata, affected file count
- Per-file breakdown: point ID, chunk index, which fields are null, page content preview, full metadata table

### Stage 2 — Delete & Re-ingest (recommended: one command)

```bash
# Auto-detect files with missing metadata, delete stale points, and re-ingest
python reingest_missing.py

# Dry run first — shows what would be detected, deleted, and re-ingested
python reingest_missing.py --dry-run

# Skip deletion (only use if points were already manually deleted)
python reingest_missing.py --skip-delete

# Manual file list (bypasses auto-detection)
python reingest_missing.py --files ../data/health_data/new_paper.pdf

# Re-ingest all files in data/health_data/ (existing points will be skipped by dedup)
python reingest_missing.py --all
```

| What it does | Queries Qdrant to find files with null metadata fields, resolves each `file_name` to a path on disk (in `../data/health_data/`), deletes the stale Qdrant points by `file_hash`, then re-ingests through the full RAGWire pipeline (load, split, LLM metadata extraction, embed, upsert). |
|---|---|
| Auto-detect | Default mode (no flags): scans Qdrant for null-metadata records and resolves their `file_name` to local file paths. No hardcoded file list. |
| `--dry-run` | Preview: shows which files have missing metadata, which paths they resolve to, and what ingestion would do — without making any changes |
| `--skip-delete` | Skip deletion of stale Qdrant points. Only use if you already ran `delete_missing_metadata_records.py` separately. Warning: RAGWire's file-level dedup will skip files whose hashes already exist. |
| `--files` | Provide explicit file paths; bypasses auto-detection |
| `--all` | Ingest all supported files in `../data/health_data/`; existing files are skipped by dedup |
| Post-ingest check | After ingestion, automatically re-queries Qdrant and reports any remaining null fields |

If a file with missing metadata is not found on disk, the script prints a warning and skips it. This handles the case where a corrupted PDF was removed from `data/health_data/`.

### Stage 2 — Delete & Re-ingest (manual, two-command alternative)

If you prefer to control deletion and ingestion separately:

```bash
# Step 2a: Delete stale points
python delete_missing_metadata_records.py --dry-run       # preview
python delete_missing_metadata_records.py                  # delete

# Step 2b: Re-ingest (skip deletion since points already removed)
python reingest_missing.py --skip-delete
```

### Stage 3 — Verify

```bash
python check_missing_metadata.py
```

Re-run the detection script. A clean result shows:

```
Total records: 75
Records with missing metadata: 0
```

---

## Configuration Files

### `health_metadata.yaml` — Metadata schema and extraction prompt

Defines what the LLM extracts from each document. Two key changes were made to fix missing metadata:

1. **`required: true`** on `publication_year` and `research_focus`

   When a field is marked `required`, the `MetadataExtractor` retries extraction with a larger document window (18,000 chars vs 5,000) if that field comes back null on the first pass. Without this flag, the retry only triggers when *all* fields are null.

2. **Enriched extraction prompt** with `## Field-Specific Extraction Hints`

   The prompt now tells the LLM exactly where to find `publication_year` (header lines, copyright notices, DOI stamps) and how to derive `research_focus` (from title, abstract, section headings).

Current schema fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | no | Full paper title as written |
| `authors` | list | no | All author names |
| `publication_year` | integer | **yes** | 4-digit year; retry triggers if null |
| `research_focus` | list | **yes** | lowercase-hyphenated topic labels; retry triggers if null |

### `config_openrouter_qdrant.yaml` — Pipeline configuration

Key settings that affect metadata extraction quality:

```yaml
llm:
  provider: "openai"
  model: "google/gemini-2.5-flash"         # Upgraded from nvidia/nemotron-3-super-120b-a12b:free
  api_key: "${OPENROUTER_API_KEY}"
  base_url: "https://openrouter.ai/api/v1"

metadata:
  config_file: "health_metadata.yaml"
```

The LLM model was upgraded from `nvidia/nemotron-3-super-120b-a12b:free` to `google/gemini-2.5-flash` for significantly better structured-output extraction accuracy.

---

## Extractor Tuning (in `ragwire/metadata/extractor.py`)

The `MetadataExtractor` class has three tunable constants that control how much document text the LLM sees:

| Constant | Original | Updated | Purpose |
|----------|----------|---------|---------|
| `PREFIX_CHARS` | 3,000 | **5,000** | Contiguous document prefix sent on first pass (title, authors, year are usually in the first page) |
| `OUTLINE_CHARS` | 1,000 | **1,500** | Cap on the markdown heading outline appended to the prefix |
| `ESCALATION_CHARS` | 12,000 | **18,000** | Window size for the second-pass retry when required fields are null |

The extraction flow:

1. **First pass** — send `PREFIX_CHARS` + heading outline (up to `OUTLINE_CHARS`) to the LLM
2. **Escalation check** — if any `required` field is null, or if *all* fields are null
3. **Retry pass** — send the first `ESCALATION_CHARS` characters of the raw document text

Increasing `PREFIX_CHARS` from 3,000 to 5,000 means the LLM sees more of the document header on the first pass, reducing the need for escalation in most cases. The larger `ESCALATION_CHARS` window gives the LLM more context for difficult documents where year or topic information appears deeper in the text.

---

## Why Deletion Before Re-ingestion

RAGWire uses **file-level deduplication**: before ingesting a file, it computes the SHA-256 hash of the file contents and checks whether any Qdrant point already carries that `file_hash`. If a match is found, the entire file is skipped.

This means you cannot simply re-ingest a file to "fix" its metadata — the pipeline will see the hash and skip it. The workflow must be:

1. **Delete** the stale points from Qdrant
2. **Re-ingest** the file so it gets fresh metadata extraction

---

## Handling Corrupted Source Files

If a PDF is a web error page rather than a real paper (e.g., a Semantic Scholar 404 page), no amount of LLM extraction will produce meaningful metadata. The report from Stage 1 will show *all* metadata fields as null for that file.

Options:
- **Replace** the PDF with a valid copy from the original source
- **Remove** the PDF from `data/health_data/` and delete its points from Qdrant — no re-ingestion needed

In this pipeline, `12_Protein_Supplementation_Review_2024.pdf` was a 404 page and was removed entirely. Its 1 Qdrant point was deleted, and no re-ingestion was performed.

---

## Acceptable Null Values

Some documents genuinely lack certain metadata. For example, `11_Hydration_Performance_2024.pdf` is a GSSI reference pamphlet with no publication date or named authors. After running the full pipeline (including the escalation retry), the result is:

- `title`: populated
- `authors`: empty list `[]` (not null — no authors exist)
- `publication_year`: **null** (no year anywhere in the document — this is correct, not a failure)
- `research_focus`: populated

A null `publication_year` is acceptable when the document genuinely does not contain one. The report from Stage 1 will still flag it, but it can be safely ignored.

---

## Quick Reference

```bash
# Recommended: full pipeline in one command
python check_missing_metadata.py              # Step 1: Detect
python reingest_missing.py --dry-run          # Step 2a: Preview what will be fixed
python reingest_missing.py                     # Step 2b: Auto-detect, delete, re-ingest, verify
python check_missing_metadata.py              # Step 3: Confirm clean

# Manual alternative: separate delete then re-ingest
python delete_missing_metadata_records.py --dry-run
python delete_missing_metadata_records.py
python reingest_missing.py --skip-delete

# Targeted re-ingestion of specific files
python reingest_missing.py --files ../data/health_data/new_paper.pdf

# Full re-ingestion of all health data (dedup skips already-ingested files)
python reingest_missing.py --all
```

All scripts must be run from the `04 Personal Gym Supplements RAG/` directory, with the virtual environment activated:

```bash
source ../.venv/bin/activate
cd "04 Personal Gym Supplements RAG/"
```