# Metadata Enhancement Recommendation

How to prevent and repair null metadata in the RAGWire ingestion pipeline.

---

## The Problem

When documents are ingested via `rag.ingest_directory('../data/health_data')`, the LLM-based metadata extractor sometimes returns null for fields like `publication_year` or `research_focus`. These nulls are written to Qdrant silently — the pipeline reports success even when every LLM-extracted field is null. Once written, the file-level dedup mechanism (`file_hash`) blocks re-ingestion, so there is no built-in path to fix the data.

The current workaround requires three external scripts (`check_missing_metadata.py`, `delete_missing_metadata_records.py`, `reingest_missing.py`) that manually detect, delete, and re-ingest. This works but is a reactive, manual process.

---

## Root Cause Analysis

### Why metadata goes null during ingestion

There are three failure modes:

1. **LLM extraction failure** — The LLM cannot find the field value in the text it was given. This has two sub-causes:
   - **Insufficient context**: The first-pass sample (5,000-char prefix + 1,500-char outline) doesn't contain the information. For example, `publication_year` appears on the first page as "Antioxidants 2025, 14, 1457" — a compact line the LLM may overlook if the prefix cuts off before it.
   - **Escalation still misses**: The retry with an 18,000-char window still doesn't contain the data, OR the LLM still fails to extract it from valid text.

2. **LLM API error** — The LLM call raises an exception (rate limit, model unavailable, structured output parse failure). The current code catches this and falls back to empty metadata (`{}`), so all LLM fields become null on every chunk.

3. **Corrupted source document** — The PDF contains garbage content (e.g., a web 404 page instead of a real paper). No extraction strategy can fix this; the file must be replaced or removed.

### Why nulls propagate silently

Looking at `_process_document()` in `pipeline.py:552-560`:

```python
llm_metadata = {}
if chunk_texts:
    try:
        llm_metadata = self.extract_metadata(text)
        logger.debug(f"LLM metadata for {file_name}: {llm_metadata}")
    except Exception as e:
        logger.warning(f"LLM metadata extraction failed for {file_name}: {e}")
```

When extraction fails or returns nulls:
- The `except` block logs a WARNING but proceeds with `llm_metadata = {}`
- There is **no check** for whether required fields came back null
- `IngestStats` has no field tracking incomplete metadata
- The file is considered "processed" and its `file_hash` is recorded, blocking future re-ingestion

### Why re-ingestion is blocked

In `ingest_documents()` (pipeline.py:413-417):

```python
file_hash = sha256_file_from_path(file_path)
if self.vectorstore_wrapper.file_hash_exists(file_hash):
    logger.info(f"Skipping (already ingested): {file_path}")
    stats["skipped"] += 1
    continue
```

The hash check is purely content-based. It cannot distinguish between "ingested with complete metadata" and "ingested with null metadata" — both produce the same `file_hash`.

---

## Enhancement Strategy

Two levels, implemented incrementally:

- **Level 1: Ingest-time resilience** — Prevent null metadata from being written silently. Catch it at ingestion time, log it clearly, and retry harder before accepting nulls.
- **Level 2: Post-ingest repair** — A built-in `repair_metadata()` method on `RAGWire` that fixes null fields in-place using Qdrant's `set_payload` API, without re-embedding.

Level 1 prevents future problems. Level 2 fixes existing ones without external scripts.

---

## Level 1: Ingest-Time Resilience

Changes to the core pipeline so that null metadata is detected, reported, and given maximum chance to succeed before being accepted.

### 1.1 Add `required_fields` validation warning in `_process_document()`

**File**: `ragwire/core/pipeline.py`
**Location**: After LLM metadata extraction (around line 560)

**Current behavior**: `extract_metadata()` returns a dict, and nulls are silently accepted.

**Change**: After extraction, check if any `required` fields from the metadata schema are still null. Log a clear WARNING per field, and record the incomplete fields in the `IngestStats` return value.

```python
# After llm_metadata is assigned (line ~555)
llm_metadata = {}
if chunk_texts:
    try:
        llm_metadata = self.extract_metadata(text)
        logger.debug(f"LLM metadata for {file_name}: {llm_metadata}")
    except Exception as e:
        logger.warning(f"LLM metadata extraction failed for {file_name}: {e}")

# NEW: Check required fields
incomplete_fields = []
if self.metadata_extractor.required_fields:
    incomplete_fields = [
        f for f in self.metadata_extractor.required_fields
        if llm_metadata.get(f) is None
    ]
    if incomplete_fields:
        logger.warning(
            f"Metadata for {file_name} has null required fields: "
            f"{', '.join(incomplete_fields)}"
        )
```

**Where to store `required_fields`**: The `MetadataExtractor` already stores `self.required_fields` (set from the YAML `required: true` annotations). The `RAGWire` instance holds `self.metadata_extractor`, so `self.metadata_extractor.required_fields` is accessible in `_process_document()`.

### 1.2 Add `incomplete_metadata` to `IngestStats`

**File**: `ragwire/core/pipeline.py`
**Location**: The `IngestStats` TypedDict (line 58-66)

**Change**: Add an `incomplete_metadata` field that lists files with null required fields:

```python
class IngestStats(TypedDict):
    total: int
    processed: int
    skipped: int
    failed: int
    chunks_created: int
    errors: List[IngestError]
    incomplete_metadata: List[dict]   # NEW: [{"file": str, "null_fields": [str]}]
```

Initialize it as `"incomplete_metadata": []` in the stats dict, and append to it when required fields are null after extraction. This gives the caller (Notebook, script, CI) a clear signal that something needs attention.

### 1.3 Add a full-document retry pass in `MetadataExtractor.extract()`

**File**: `ragwire/metadata/extractor.py`
**Location**: After the existing escalation retry (around line 203-210)

**Current behavior**: One retry with `ESCALATION_CHARS` (18,000 chars). If that still has null required fields, the nulls are accepted.

**Change**: Add a third pass that sends the **entire document text** (no truncation) as a last resort. This is expensive but rare — it only fires when the 18k-char window still missed required fields.

```python
# After the existing escalation retry block...
if self._needs_escalation(metadata) and len(text) > len(sample):
    logger.info("Metadata extraction missed fields — retrying with larger window")
    retry = chain.invoke({"content": text[: self.ESCALATION_CHARS]}).model_dump()
    metadata = {
        k: retry.get(k) if retry.get(k) is not None else v
        for k, v in metadata.items()
    }

# NEW: Third pass — full document text, only if still missing required fields
if self._needs_escalation(metadata) and len(text) > self.ESCALATION_CHARS:
    logger.info("Metadata still incomplete — retrying with full document text")
    full_retry = chain.invoke({"content": text}).model_dump()
    metadata = {
        k: full_retry.get(k) if full_retry.get(k) is not None else v
        for k, v in metadata.items()
    }
```

This adds a full-document pass as the final attempt. The cost is one additional LLM call per document, but only for documents where required fields are still null after the 18k-char retry. In practice this will be rare.

### 1.4 Expose extraction diagnostics on chunk metadata (optional)

**File**: `ragwire/core/pipeline.py`
**Location**: The chunk_metadata dict in `_process_document()` (around line 571-582)

**Change**: Add an `_extraction_pass` field that records which attempt produced the final metadata:

```python
chunk_metadata = {
    "source": file_path,
    "file_name": file_name,
    ...
    **llm_metadata,
    "_extraction_pass": extraction_pass,  # NEW: 1=prefix, 2=escalation, 3=full
}
```

This is useful for debugging but not essential. Implement only if there's a need to audit extraction quality.

---

## Level 2: Post-Ingest Repair

A new `repair_metadata()` method on `RAGWire` that fixes null metadata fields in-place using Qdrant's `set_payload` API, without re-embedding or re-chunking.

### 2.1 How `set_payload` works

Qdrant's `client.set_payload()` merges key-value pairs into the existing payload of specified points. It does **not** replace the entire payload — only the specified keys are updated. This means we can update `metadata.publication_year` from null to `2025` without touching vectors, embeddings, or any other metadata fields.

Key capability from the Qdrant Python client:

```python
from qdrant_client.http import models as rest

client.set_payload(
    collection_name="health-rag-bge-m3-qdrant",
    payload={"metadata.publication_year": 2025, "metadata.research_focus": ["oxidative-stress"]},
    points=rest.Filter(
        must=[rest.FieldCondition(key="metadata.file_hash", match=rest.MatchValue(value="abc123..."))]
    ),
)
```

This updates only the specified fields on all points matching the filter. No re-embedding, no point ID changes, no data loss.

### 2.2 New method: `QdrantStore.update_metadata_by_file_hash()`

**File**: `ragwire/vectorstores/qdrant_store.py`
**Location**: After `file_hash_exists()` (around line 244)

```python
def update_metadata_by_file_hash(
    self,
    file_hash: str,
    metadata_updates: dict,
) -> int:
    """
    Update metadata fields on all points belonging to a file.

    Uses Qdrant's set_payload to merge updates into existing point payloads.
    Does NOT touch vectors or embeddings. Only updates the specified fields.

    Args:
        file_hash: SHA256 hash of the file whose points should be updated
        metadata_updates: Dict of metadata fields to update, e.g.
            {"publication_year": 2025, "research_focus": ["oxidative-stress"]}
            Keys should NOT include the "metadata." prefix.

    Returns:
        Number of points updated
    """
    from qdrant_client.http import models as rest

    if not self.collection_name or not self.collection_exists():
        raise ValueError(f"Collection '{self.collection_name}' does not exist")

    # Build the payload dict with "metadata." prefix for Qdrant storage
    payload = {}
    for key, value in metadata_updates.items():
        payload[f"metadata.{key}"] = value

    # Find all point IDs for this file
    points, _ = self.client.scroll(
        collection_name=self.collection_name,
        scroll_filter=rest.Filter(
            must=[rest.FieldCondition(
                key="metadata.file_hash",
                match=rest.MatchValue(value=file_hash),
            )]
        ),
        limit=1000,
        with_payload=False,
        with_vectors=False,
    )

    if not points:
        return 0

    point_ids = [p.id for p in points]

    # set_payload merges into existing payload — does not overwrite other fields
    self.client.set_payload(
        collection_name=self.collection_name,
        payload=payload,
        points=point_ids,
    )

    logger.info(f"Updated {len(point_ids)} points for file_hash={file_hash[:16]}... with fields: {list(metadata_updates.keys())}")
    return len(point_ids)
```

### 2.3 New method: `QdrantStore.find_incomplete_metadata()`

**File**: `ragwire/vectorstores/qdrant_store.py`
**Location**: After `get_field_values()` (around line 340)

```python
def find_incomplete_metadata(
    self,
    required_fields: list[str],
) -> dict[str, dict]:
    """
    Find files that have null values in any required metadata field.

    Scans all points in the collection and groups results by file_hash.

    Args:
        required_fields: Metadata field names that must not be null, e.g.
            ["title", "authors", "publication_year", "research_focus"]

    Returns:
        Dict mapping file_hash -> {
            "file_name": str,
            "null_fields": set of field names that are null,
            "point_count": int,
        }
    """
    from qdrant_client.http import models as rest

    file_hashes_with_nulls = {}
    offset = None

    while True:
        points, offset = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            meta = payload.get("metadata", payload)
            if meta is None:
                continue

            null_fields = [f for f in required_fields if meta.get(f) is None]
            if null_fields:
                file_hash = meta.get("file_hash", "")
                file_name = meta.get("file_name", "unknown")
                if file_hash not in file_hashes_with_nulls:
                    file_hashes_with_nulls[file_hash] = {
                        "file_name": file_name,
                        "null_fields": set(null_fields),
                        "point_count": 0,
                    }
                else:
                    file_hashes_with_nulls[file_hash]["null_fields"].update(null_fields)
                file_hashes_with_nulls[file_hash]["point_count"] += 1

        if offset is None:
            break

    return file_hashes_with_nulls
```

### 2.4 New method: `RAGWire.repair_metadata()`

**File**: `ragwire/core/pipeline.py`
**Location**: After `get_stats()` method (around line 976)

This is the main public API. It:
1. Queries Qdrant for files with null required fields
2. Loads the source document from disk
3. Re-runs metadata extraction with the current (possibly upgraded) LLM
4. Updates only the null fields via `set_payload` — no re-embedding

```python
def repair_metadata(
    self,
    data_dir: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Find and repair null metadata fields in the vector store.

    Scans the collection for points with null required fields, loads the
    source documents, re-extracts metadata using the configured LLM, and
    updates only the null fields in-place using Qdrant's set_payload API.

    This does NOT re-embed or re-chunk. It is a lightweight metadata-only
    repair that preserves vectors and point IDs.

    Args:
        data_dir: Directory to search for source files. If None, looks for
            files relative to the "source" metadata field stored in Qdrant.
        dry_run: If True, report what would be repaired without making changes.

    Returns:
        Dict with keys:
            - "scanned": total points scanned
            - "files_with_nulls": number of files with null required fields
            - "repaired": number of files successfully repaired
            - "unrepairable": list of files that couldn't be repaired
              (source file not found, or re-extraction still null)
            - "updated_points": total number of Qdrant points updated
    """
    required = self.metadata_extractor.required_fields or self._filter_fields
    if not required:
        logger.warning("No required fields configured — nothing to repair")
        return {
            "scanned": 0, "files_with_nulls": 0, "repaired": 0,
            "unrepairable": [], "updated_points": 0,
        }

    # Step 1: Find files with null metadata
    incomplete = self.vectorstore_wrapper.find_incomplete_metadata(required)

    if not incomplete:
        logger.info("No files with incomplete metadata found — collection is clean!")
        return {
            "scanned": 0, "files_with_nulls": 0, "repaired": 0,
            "unrepairable": [], "updated_points": 0,
        }

    logger.info(f"Found {len(incomplete)} file(s) with incomplete metadata")

    results = {
        "scanned": 0,
        "files_with_nulls": len(incomplete),
        "repaired": 0,
        "unrepairable": [],
        "updated_points": 0,
    }

    for file_hash, info in incomplete.items():
        file_name = info["file_name"]
        null_fields = sorted(info["null_fields"])

        # Step 2: Locate source file
        source_path = self._resolve_source_path(file_name, data_dir)
        if source_path is None:
            logger.warning(f"Source file not found for {file_name} — skipping repair")
            results["unrepairable"].append({
                "file_name": file_name,
                "null_fields": null_fields,
                "reason": "source file not found",
            })
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would repair {file_name}: {', '.join(null_fields)}")
            continue

        # Step 3: Load and re-extract metadata
        load_result = self.loader.load(str(source_path))
        if not load_result["success"]:
            results["unrepairable"].append({
                "file_name": file_name,
                "null_fields": null_fields,
                "reason": f"load failed: {load_result['error']}",
            })
            continue

        try:
            new_metadata = self.extract_metadata(load_result["text_content"])
        except Exception as e:
            results["unrepairable"].append({
                "file_name": file_name,
                "null_fields": null_fields,
                "reason": f"extraction failed: {e}",
            })
            continue

        # Step 4: Pick only the fields that were null and are now populated
        updates = {}
        still_null = []
        for field in null_fields:
            if new_metadata.get(field) is not None:
                updates[field] = new_metadata[field]
            else:
                still_null.append(field)

        if not updates:
            logger.warning(f"Re-extraction still null for {file_name}: {', '.join(still_null)}")
            results["unrepairable"].append({
                "file_name": file_name,
                "null_fields": still_null,
                "reason": "re-extraction still null",
            })
            continue

        # Step 5: Update metadata in Qdrant (no re-embedding)
        point_count = self.vectorstore_wrapper.update_metadata_by_file_hash(
            file_hash, updates
        )
        results["repaired"] += 1
        results["updated_points"] += point_count
        logger.info(
            f"Repaired {file_name}: updated {list(updates.keys())} "
            f"on {point_count} points"
            + (f" (still null: {', '.join(still_null)})" if still_null else "")
        )

    # Step 6: Refresh payload indexes and stored-values cache
    all_fields = self.vectorstore_wrapper.get_metadata_keys()
    self.vectorstore_wrapper.create_payload_indexes(all_fields)
    self._stored_values_cache = None

    return results


def _resolve_source_path(self, file_name: str, data_dir: Optional[str] = None) -> Optional[Path]:
    """
    Try to find a source file on disk by its name.

    Checks the data_dir (if provided), then the source paths stored
    in Qdrant metadata, then common data directories.

    Returns Path if found, None otherwise.
    """
    from pathlib import Path

    # Try data_dir first
    if data_dir:
        candidate = Path(data_dir) / file_name
        if candidate.exists():
            return candidate

    # Try to find source path from Qdrant metadata
    # (stored as "source" field, e.g. "../data/health_data/file.pdf")
    results, _ = self.vectorstore_wrapper.client.scroll(
        collection_name=self.vectorstore_wrapper.collection_name,
        scroll_filter=None,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    # Search by file_name in all points' metadata
    from qdrant_client.http import models as rest
    results, _ = self.vectorstore_wrapper.client.scroll(
        collection_name=self.vectorstore_wrapper.collection_name,
        scroll_filter=rest.Filter(
            must=[rest.FieldCondition(
                key="metadata.file_name",
                match=rest.MatchValue(value=file_name),
            )]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if results:
        source = (results[0].payload or {}).get("metadata", {}).get("source", "")
        candidate = Path(source)
        if candidate.exists():
            return candidate

    # Try common data directories relative to project root
    for search_dir in ["../data/health_data", "data/health_data", "data"]:
        candidate = Path(search_dir) / file_name
        if candidate.exists():
            return candidate

    return None
```

### 2.5 Expose `repair_metadata` in the public API

**File**: `ragwire/__init__.py`

Add `repair_metadata` to the `RAGWire` class — it's already a method on the class, so it's automatically available. No `__init__.py` changes needed since `RAGWire` is already exported.

### 2.6 Add `find_incomplete_metadata` and `update_metadata_by_file_hash` to public API (optional)

If these utility methods should be callable independently:

**File**: `ragwire/__init__.py`

```python
# These are accessible via rag.vectorstore_wrapper.find_incomplete_metadata(...)
# No new exports needed — they're methods on QdrantStore which is already exported.
```

No changes needed — `QdrantStore` is already in `__all__`.

---

## Implementation Order

### Phase 1: Ingest-time resilience (prevents future problems)

| Step | File | Change | Effort |
|------|------|--------|--------|
| 1.1 | `ragware/core/pipeline.py` | Add `incomplete_metadata` to `IngestStats` | Small |
| 1.2 | `ragware/core/pipeline.py` | Check `required_fields` after extraction, log WARNING, append to stats | Small |
| 1.3 | `ragware/metadata/extractor.py` | Add full-document retry pass (3rd attempt) | Small |
| 1.4 | `ragware/core/pipeline.py` | (Optional) Add `_extraction_pass` field to chunk metadata | Small |

**Test criteria**: Ingest a document that previously produced null `publication_year` or `research_focus`. Verify in logs that the 3rd-pass retry fires when needed, and `IngestStats` contains `incomplete_metadata` entries for files where fields are still null.

### Phase 2: Post-ingest repair (fixes existing problems)

| Step | File | Change | Effort |
|------|------|--------|--------|
| 2.1 | `ragware/vectorstores/qdrant_store.py` | Add `find_incomplete_metadata()` method | Medium |
| 2.2 | `ragware/vectorstores/qdrant_store.py` | Add `update_metadata_by_file_hash()` method | Medium |
| 2.3 | `ragware/core/pipeline.py` | Add `repair_metadata()` method | Medium |
| 2.4 | `ragware/core/pipeline.py` | Add `_resolve_source_path()` helper | Small |

**Test criteria**: 
1. Run `rag.repair_metadata(data_dir="../data/health_data")` on the current collection where file 11 (`11_Hydration_Performance_2024.pdf`) has `publication_year=null`.
2. Verify the method finds the null field, locates the source file, re-extracts metadata, and updates only the null field via `set_payload`.
3. Run `rag.repair_metadata(dry_run=True)` and confirm it reports what would be changed without making changes.

### Phase 3: Script cleanup (optional)

Once Phase 2 is implemented, the external scripts become less critical:
- `check_missing_metadata.py` — still useful as a diagnostic report tool (writes Markdown); does not need to change
- `reingest_missing.py` — can be simplified to call `rag.repair_metadata()` internally instead of the delete-then-reingest cycle for metadata-only fixes. The `--all` and `--files` flags still serve a purpose for full re-ingestion scenarios (corrupted PDFs, changed chunking strategy)
- `delete_missing_metadata_records.py` — still needed for manual cleanup when a file should be completely removed from the collection

---

## Design Decisions and Trade-offs

### Why `set_payload` instead of delete + re-ingest?

| Aspect | `set_payload` (repair_metadata) | Delete + re-ingest |
|--------|----------------------------------|---------------------|
| **Speed** | Fast — no embedding computation | Slow — must embed every chunk |
| **Cost** | 1 LLM call per file for metadata | LLM call + embedding for every chunk |
| **Point IDs** | Preserved | New UUIDs assigned |
| **Vectors** | Untouched | Re-computed (may differ due to model version) |
| **Chunk text** | Untouched | Re-split (may differ if splitter config changed) |
| **Use case** | Fix null metadata fields | Re-process from scratch (new chunking, new embeddings) |

For pure metadata fixes, `set_payload` is strictly better. Re-ingestion is appropriate when the source document or processing pipeline has changed.

### Why a 3rd retry with the full document?

The current 2-pass strategy (5k prefix, then 18k window) fails in two scenarios:
1. The key info is beyond the 18k-char window (unlikely for typical papers, but possible for very long documents)
2. The LLM simply missed it in the window — a model-quality issue, not a window-size issue

The 3rd pass sends the entire document to the LLM. It's expensive (one more API call, maximum token consumption) but it eliminates window truncation as a cause. It only fires when required fields are still null after the 18k retry, so it won't affect the common case.

### Why not increase `ESCALATION_CHARS` further instead?

`ESCALATION_CHARS` at 18,000 already covers most documents. Going beyond 18k increases token cost for every failed first-pass extraction, and many LLMs struggle with very long contexts for structured extraction. The 3rd-pass approach is better because:
- It only fires when the 18k retry also fails (rare)
- It sends the full document, guaranteeing the info is present
- It can use a different prompt strategy if desired (future enhancement)

### What about corrupted source files (404 pages)?

`repair_metadata()` cannot fix corrupted PDFs — no amount of LLM extraction will produce metadata from a Semantic Scholar 404 page. The method handles this gracefully:
- `_resolve_source_path()` locates the source file on disk
- `loader.load()` may succeed (the file exists) but the content is garbage
- `extract_metadata()` will likely return null for all fields
- The method reports the file as "unrepairable: re-extraction still null"
- The user must replace or remove the source file manually

This is the correct behavior — the method can't solve data quality problems, only extraction quality problems.

---

## Key Files Reference

| File | Role |
|------|------|
| `ragware/core/pipeline.py` | Main `RAGWire` class — ingestion, retrieval, `repair_metadata()` (new) |
| `ragware/core/pipeline.py:52-66` | `IngestStats` TypedDict — needs `incomplete_metadata` field |
| `ragware/core/pipeline.py:370-469` | `ingest_documents()` — needs required-field validation |
| `ragware/core/pipeline.py:519-586` | `_process_document()` — needs extraction-pass tracking |
| `ragware/metadata/extractor.py:89-92` | Window size constants — currently 5000/1500/18000 |
| `ragware/metadata/extractor.py:162-220` | `extract()` — needs 3rd-pass full-document retry |
| `ragware/metadata/extractor.py:152-160` | `_needs_escalation()` — logic that triggers retries |
| `ragware/vectorstores/qdrant_store.py` | `QdrantStore` — needs `find_incomplete_metadata()` and `update_metadata_by_file_hash()` |
| `health_metadata.yaml` | Schema config with `required: true` annotations |
| `config_openrouter_qdrant.yaml` | Pipeline config (LLM model, embedding model, Qdrant connection) |

---

## Usage After Implementation

### Phase 1 — Prevent future nulls during ingestion

```python
from ragwire import RAGWire

rag = RAGWire("config_openrouter_qdrant.yaml")
stats = rag.ingest_directory("../data/health_data")

# Check if any files had incomplete metadata
if stats["incomplete_metadata"]:
    print(f"Warning: {len(stats['incomplete_metadata'])} files have null required fields:")
    for entry in stats["incomplete_metadata"]:
        print(f"  {entry['file']}: null fields = {entry['null_fields']}")
```

### Phase 2 — Repair existing nulls without re-ingesting

```python
from ragwire import RAGWire

rag = RAGWire("config_openrouter_qdrant.yaml")

# Dry run — see what would be fixed
result = rag.repair_metadata(data_dir="../data/health_data", dry_run=True)
print(result)

# Actual repair
result = rag.repair_metadata(data_dir="../data/health_data")
print(f"Repaired {result['repaired']} files, updated {result['updated_points']} points")
for entry in result["unrepairable"]:
    print(f"  Could not fix {entry['file_name']}: {entry['reason']}")
```

### External scripts — still useful for diagnostics and full re-ingestion

```bash
# Diagnostic report (still the best way to see full details of every null field)
python check_missing_metadata.py

# Full re-ingestion (needed when chunking strategy or embedding model changes)
python reingest_missing.py --all

# Manual deletion (needed when removing a file entirely)
python delete_missing_metadata_records.py --dry-run
```