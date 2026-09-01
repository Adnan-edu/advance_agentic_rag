#!/usr/bin/env python3
"""
Cleanup script for Hugging Face model cache.

Default:  remove only the BAAI/bge-m3 weights this project downloaded.
--all:    list every cached HF model/dataset/space and remove them (asks first).
--list:   just show what is currently cached, do not delete anything.
--yes:    skip the confirmation prompt (CI / scripted use).

Uses huggingface_hub's official cache API so reference symlinks and blobs are
removed atomically — never `rm -rf` on the cache directory.

Run from the repository root:
    uv run python scripts/cleanup_hf_cache.py          # remove BAAI/bge-m3
    uv run python scripts/cleanup_hf_cache.py --list   # inspect first
    uv run python scripts/cleanup_hf_cache.py --all    # nuke everything cached
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable

try:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.utils import CacheNotFound
except ImportError:
    sys.stderr.write(
        "huggingface_hub is not installed in this environment.\n"
        "Install the locked project environment first:\n"
        "  uv sync\n"
    )
    sys.exit(1)


DEFAULT_TARGET = "BAAI/bge-m3"


def human_bytes(n: int) -> str:
    """Format byte counts the way the HF CLI does."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < step:
            return f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} PB"


def list_cache() -> None:
    """Print every cached repo with its on-disk size and last-modified time."""
    try:
        info = scan_cache_dir()
    except CacheNotFound:
        print("No Hugging Face cache directory found — nothing to list.")
        return

    if not info.repos:
        print(f"Cache at {info.size_on_disk_str} but no repos found.")
        return

    print(f"Cache root: {info.repos.__iter__().__next__().repo_path.parent}")
    print(f"Total cached: {info.size_on_disk_str} across {len(info.repos)} repo(s)\n")

    for repo in sorted(info.repos, key=lambda r: r.size_on_disk, reverse=True):
        print(
            f"  {repo.repo_type:8s} {repo.repo_id:60s} "
            f"{human_bytes(repo.size_on_disk):>10s}"
        )


def revisions_for(repo_ids: Iterable[str]) -> list[str]:
    """Resolve repo_ids → all revision hashes in cache, warn on misses."""
    try:
        info = scan_cache_dir()
    except CacheNotFound:
        print("No Hugging Face cache directory found — nothing to delete.")
        return []

    wanted = {rid for rid in repo_ids}
    matched: set[str] = set()
    revisions: list[str] = []

    for repo in info.repos:
        if repo.repo_id in wanted:
            matched.add(repo.repo_id)
            revisions.extend(rev.commit_hash for rev in repo.revisions)

    missing = wanted - matched
    for rid in sorted(missing):
        print(f"  (not in cache, skipping): {rid}")

    return revisions


def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


def delete(repo_ids: list[str], assume_yes: bool) -> int:
    """Delete the given repo_ids from the cache. Returns process exit code."""
    revisions = revisions_for(repo_ids)
    if not revisions:
        print("Nothing to delete.")
        return 0

    info = scan_cache_dir()
    strategy = info.delete_revisions(*revisions)

    pretty = ", ".join(repo_ids)
    print(
        f"\nAbout to free {strategy.expected_freed_size_str} "
        f"by removing {len(revisions)} revision(s) from: {pretty}"
    )

    if not confirm("Proceed?", assume_yes):
        print("Aborted.")
        return 1

    strategy.execute()
    print(f"Done. Freed {strategy.expected_freed_size_str}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list", action="store_true", help="List the cache, do not delete."
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Delete every cached repo (asks before deleting).",
    )
    group.add_argument(
        "--repo",
        action="append",
        default=None,
        metavar="REPO_ID",
        help=(
            "Specific repo_id to delete (e.g. BAAI/bge-m3). "
            "May be passed multiple times. Defaults to BAAI/bge-m3."
        ),
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip the confirmation prompt."
    )
    args = parser.parse_args()

    if args.list:
        list_cache()
        return 0

    if args.all:
        try:
            info = scan_cache_dir()
        except CacheNotFound:
            print("No Hugging Face cache directory found — nothing to delete.")
            return 0
        repos = [r.repo_id for r in info.repos]
        if not repos:
            print("Cache is already empty.")
            return 0
        print("Will delete every cached repo:")
        for rid in repos:
            print(f"  - {rid}")
        return delete(repos, assume_yes=args.yes)

    targets = args.repo or [DEFAULT_TARGET]
    return delete(targets, assume_yes=args.yes)


if __name__ == "__main__":
    sys.exit(main())
