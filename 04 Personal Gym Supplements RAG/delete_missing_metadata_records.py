"""
Delete records from Qdrant that have missing metadata, so they can be re-ingested.

Usage:
    python delete_missing_metadata_records.py                  # Delete all files with missing metadata
    python delete_missing_metadata_records.py --file-hashes HASH1 HASH2   # Delete specific file hashes
    python delete_missing_metadata_records.py --dry-run          # Preview without deleting

Reads the Qdrant connection from config_openrouter_qdrant.yaml and uses
the same environment variable resolution as the ragwire pipeline.
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

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config_openrouter_qdrant.yaml"


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


def find_missing_metadata_file_hashes(client: QdrantClient, collection: str) -> dict[str, list[str]]:
    """Scan the collection and return {file_hash: [null_field_names]} for files with null metadata."""
    metadata_fields = ["title", "authors", "publication_year", "research_focus"]
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

            null_fields = [f for f in metadata_fields if meta.get(f) is None]
            if null_fields:
                file_hash = meta.get("file_hash", "")
                file_name = meta.get("file_name", "unknown")
                if file_hash not in file_hashes_with_nulls:
                    file_hashes_with_nulls[file_hash] = {
                        "file_name": file_name,
                        "null_fields": set(null_fields),
                    }
                else:
                    file_hashes_with_nulls[file_hash]["null_fields"].update(null_fields)

        if offset is None:
            break

    return file_hashes_with_nulls


def count_points_by_hash(client: QdrantClient, collection: str, file_hash: str) -> int:
    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=rest.Filter(
            must=[rest.FieldCondition(
                key="metadata.file_hash",
                match=rest.MatchValue(value=file_hash),
            )]
        ),
        limit=100,
        with_payload=False,
        with_vectors=False,
    )
    return len(results)


def delete_by_file_hashes(client: QdrantClient, collection: str, file_hashes: list[str], file_names: dict[str, str] | None = None, dry_run: bool = False):
    """Delete all points matching the given file hashes."""
    total_points = 0
    for fh in file_hashes:
        count = count_points_by_hash(client, collection, fh)
        name = (file_names or {}).get(fh, fh[:16])
        if dry_run:
            print(f"  [DRY RUN] Would delete {count} points for {name} (hash: {fh[:24]}...)")
        else:
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
            print(f"  Deleted {count} points for {name} (hash: {fh[:24]}...)")
        total_points += count

    if not dry_run:
        info = client.get_collection(collection)
        print(f"\nCollection total points after deletion: {info.points_count}")
    else:
        print(f"\n[DRY RUN] Total points that would be deleted: {total_points}")

    return total_points


def main():
    parser = argparse.ArgumentParser(description="Delete Qdrant records with missing metadata")
    parser.add_argument("--file-hashes", nargs="+", help="Specific file hashes to delete")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be deleted without making changes")
    args = parser.parse_args()

    client, collection = get_client_and_collection(CONFIG_PATH)

    if args.file_hashes:
        file_hashes = args.file_hashes
        file_names = {}
    else:
        print("Scanning collection for files with missing metadata...")
        file_hashes_with_nulls = find_missing_metadata_file_hashes(client, collection)

        if not file_hashes_with_nulls:
            print("No records with missing metadata found. Collection is clean!")
            return

        print(f"Found {len(file_hashes_with_nulls)} file(s) with missing metadata:\n")
        file_hashes = []
        file_names = {}
        for fh, info in file_hashes_with_nulls.items():
            print(f"  {info['file_name']}: missing {', '.join(sorted(info['null_fields']))} ({count_points_by_hash(client, collection, fh)} points)")
            file_hashes.append(fh)
            file_names[fh] = info["file_name"]

        print()

    total = delete_by_file_hashes(client, collection, file_hashes, file_names=file_names, dry_run=args.dry_run)

    if not args.dry_run and file_hashes:
        print("\nVerifying deletion...")
        remaining = 0
        for fh in file_hashes:
            count = count_points_by_hash(client, collection, fh)
            name = (file_names or {}).get(fh, fh[:16])
            if count > 0:
                print(f"  WARNING: {name} still has {count} points remaining!")
                remaining += count
        if remaining == 0:
            print("  All target points successfully deleted.")


if __name__ == "__main__":
    main()
