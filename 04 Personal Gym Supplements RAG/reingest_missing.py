"""
Re-ingest files with missing metadata into Qdrant via the RAGWire pipeline.

Automatically detects files with null metadata fields by querying Qdrant,
resolves them to file paths on disk, deletes the stale points, and re-ingests
them with fresh metadata extraction.

Usage:
    python reingest_missing.py                  # Auto-detect & re-ingest files with missing metadata
    python reingest_missing.py --dry-run        # Preview what would be done
    python reingest_missing.py --skip-delete     # Re-ingest without deleting first (file_hash must not exist)
    python reingest_missing.py --files path/to/file1.pdf path/to/file2.pdf
    python reingest_missing.py --all             # Re-ingest all files in data/health_data/
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv(override=True)

# Running this file by path places Lesson 04 before the repository root on
# sys.path; select the canonical package instead of the stale lesson copy.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ragwire import RAGWire, setup_logging

CONFIG_PATH = Path(__file__).parent / "config_openrouter_qdrant.yaml"
DATA_DIR = Path(__file__).parent.parent / "data" / "health_data"

METADATA_FIELDS = ["title", "authors", "publication_year", "research_focus"]
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"}


def resolve_env(value: str) -> str:
    match = re.match(r"\$\{(.+)\}", value)
    if match:
        return os.getenv(match.group(1), "")
    return value


def get_client_and_collection(config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    vs = config["vectorstore"]
    client = QdrantClient(
        url=resolve_env(vs["url"]),
        api_key=resolve_env(vs.get("api_key", "")),
    )
    collection = vs["collection_name"]
    return client, collection


def selected_model(config_path: Path, model_tier: str) -> str:
    """Resolve and validate the configured model without initializing Qdrant."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    llm = config.get("llm", {})
    options = llm.get("model_options")
    if not options:
        raise ValueError(
            "--model-tier requires llm.model_options in the configuration"
        )
    if model_tier not in options:
        raise ValueError(
            f"Invalid model tier {model_tier!r}; choose one of: {', '.join(options)}"
        )
    return options[model_tier]


def find_missing_metadata_files(client: QdrantClient, collection: str) -> dict[str, dict]:
    """Scan Qdrant for files whose chunks have null metadata fields.

    Returns:
        {file_hash: {"file_name": str, "null_fields": set, "point_count": int}}
    """
    file_hashes_with_nulls = {}
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
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

            null_fields = [f for f in METADATA_FIELDS if meta.get(f) is None]
            if null_fields:
                file_hash = meta.get("file_hash", "")
                file_name = meta.get("file_name", "unknown")
                if file_hash not in file_hashes_with_nulls:
                    file_hashes_with_nulls[file_hash] = {
                        "file_name": file_name,
                        "null_fields": set(null_fields),
                        "point_count": 0,
                    }
                file_hashes_with_nulls[file_hash]["null_fields"].update(null_fields)
                file_hashes_with_nulls[file_hash]["point_count"] += 1

        if offset is None:
            break

    return file_hashes_with_nulls


def resolve_file_path(file_name: str) -> Path | None:
    """Find a file on disk by its name, checking the data directory."""
    candidate = DATA_DIR / file_name
    if candidate.exists():
        return candidate
    return None


def delete_stale_points(client: QdrantClient, collection: str, file_hashes: list[str], file_names: dict[str, str]) -> int:
    """Delete all Qdrant points belonging to the given file hashes."""
    total_deleted = 0
    for fh in file_hashes:
        name = file_names.get(fh, fh[:16])
        client.delete(
            collection_name=collection,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[rest.FieldCondition(
                        key="metadata.file_hash",
                        match=rest.MatchValue(value=fh),
                    )]
                )
            ),
        )
        count_info = client.get_collection(collection)
        print(f"  Deleted stale points for {name} (collection now has {count_info.points_count} points)")
        total_deleted += 1
    return total_deleted


def collect_all_files(data_dir: Path) -> list[Path]:
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(
        description="Re-ingest files with missing metadata (auto-detect, delete stale points, re-ingest)"
    )
    parser.add_argument("--files", nargs="+", help="Specific file paths to ingest (skips auto-detection)")
    parser.add_argument("--all", action="store_true", help="Ingest all files in data/health_data/")
    parser.add_argument("--skip-delete", action="store_true", help="Skip deleting stale Qdrant points before re-ingesting")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be done without making changes")
    parser.add_argument(
        "--model-tier",
        choices=("free", "paid"),
        default=os.getenv("RAGWIRE_MODEL_TIER", "free"),
        help="LLM tier for metadata extraction (default: RAGWIRE_MODEL_TIER or free)",
    )
    args = parser.parse_args()
    model_id = selected_model(CONFIG_PATH, args.model_tier)

    # ── Determine which files to ingest ──────────────────────────────────

    if args.files:
        file_paths = [Path(f) for f in args.files]
        source = "manual"
    elif args.all:
        file_paths = collect_all_files(DATA_DIR)
        source = "--all"
    else:
        # Auto-detect: query Qdrant for files with null metadata fields,
        # then resolve those file names to paths on disk.
        print("Scanning Qdrant for files with missing metadata...\n")
        client, collection = get_client_and_collection(CONFIG_PATH)
        missing_files = find_missing_metadata_files(client, collection)

        if not missing_files:
            print("No records with missing metadata found. Collection is clean!")
            return

        print(f"Found {len(missing_files)} file(s) with missing metadata:\n")
        file_paths = []
        file_hashes_to_delete = []
        file_names = {}

        for fh, info in missing_files.items():
            file_name = info["file_name"]
            null_fields = sorted(info["null_fields"])
            point_count = info["point_count"]
            print(f"  {file_name}: missing {', '.join(null_fields)} ({point_count} points)")

            resolved = resolve_file_path(file_name)
            if resolved:
                file_paths.append(resolved)
                file_hashes_to_delete.append(fh)
                file_names[fh] = file_name
                print(f"    -> resolved to {resolved}")
            else:
                on_disk = DATA_DIR / file_name
                print(f"    -> WARNING: file not found at {on_disk}, skipping re-ingestion")
                print(f"       To fix: place the file in {DATA_DIR} and re-run this script")

        if not file_paths:
            print("\nNo resolvable files found on disk. Nothing to re-ingest.")
            return

        source = "auto-detect"

        # ── Delete stale points before re-ingesting ───────────────────────
        print(f"\nSelected {args.model_tier} model: {model_id}")
        if not args.skip_delete:
            if args.dry_run:
                print(f"\n[DRY RUN] Would delete {len(file_hashes_to_delete)} file(s) from Qdrant:")
            else:
                print(f"\nDeleting stale Qdrant points for {len(file_hashes_to_delete)} file(s)...")
                delete_stale_points(client, collection, file_hashes_to_delete, file_names)
        else:
            print("\nSkipping deletion (--skip-delete). File-level dedup may cause files to be skipped.")

    # ── Validate all files exist on disk ─────────────────────────────────

    missing_on_disk = [f for f in file_paths if not f.exists()]
    if missing_on_disk:
        print("\nERROR: The following files do not exist:")
        for f in missing_on_disk:
            print(f"  {f}")
        sys.exit(1)

    file_paths_str = [str(f) for f in file_paths]

    print(f"\nFiles to ingest ({len(file_paths_str)}, source: {source}):")
    for f in file_paths_str:
        print(f"  {f}")

    print(f"Selected {args.model_tier} model: {model_id}")

    if args.dry_run:
        print("\n[DRY RUN] No ingestion performed.")
        return

    # ── Ingest through the RAGWire pipeline ───────────────────────────────

    setup_logging(log_level="INFO")
    rag = RAGWire(str(CONFIG_PATH), model_tier=args.model_tier)

    stats = rag.ingest_documents(file_paths_str)

    print(f"\n{'='*40}")
    print("Ingestion Results")
    print(f"{'='*40}")
    print(f"  Total:          {stats['total']}")
    print(f"  Processed:      {stats['processed']}")
    print(f"  Skipped:        {stats['skipped']}")
    print(f"  Failed:         {stats['failed']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    if stats["errors"]:
        print(f"\nErrors:")
        for err in stats["errors"]:
            print(f"  {err['file']}: {err['error']}")

    # ── Post-ingest verification ──────────────────────────────────────────
    print(f"\n{'='*40}")
    print("Post-ingest verification")
    print(f"{'='*40}")
    client, collection = get_client_and_collection(CONFIG_PATH)
    missing_after = find_missing_metadata_files(client, collection)
    if not missing_after:
        print("  All records have complete metadata.")
    else:
        for fh, info in missing_after.items():
            print(f"  {info['file_name']}: still missing {', '.join(sorted(info['null_fields']))}")
        print(f"\n  {len(missing_after)} file(s) still have null fields.")
        print("  This may be acceptable if the source document genuinely lacks that information.")


if __name__ == "__main__":
    main()
