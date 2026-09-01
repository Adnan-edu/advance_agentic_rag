#!/usr/bin/env python3
"""Quick health probe for local Qdrant or the configured cloud cluster.

Local mode is the default so this command verifies the service started by
``start_qdrant.sh``. Cloud mode reads QDRANT_URL and QDRANT_API_KEY from the
repository-root ``.env`` file.

Usage:
    uv run scripts/check_qdrant.py          # http://localhost:6333
    uv run scripts/check_qdrant.py cloud    # QDRANT_URL from .env
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Resolve from this file so the script works regardless of the current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=("local", "cloud"),
        nargs="?",
        default="local",
        help="Endpoint to verify (default: local)",
    )
    return parser.parse_args()


def connection_settings(target: str) -> tuple[str, dict[str, str]]:
    if target == "local":
        return "http://localhost:6333", {}

    url = os.environ.get("QDRANT_URL", "").rstrip("/")
    key = os.environ.get("QDRANT_API_KEY", "")
    if not url or not key:
        sys.exit("Missing QDRANT_URL or QDRANT_API_KEY in the repository .env")
    return url, {"api-key": key}


def main() -> int:
    args = parse_args()
    url, headers = connection_settings(args.target)
    print(f"Probing {args.target} Qdrant at {url} ...")

    ok = True
    for path in ("/healthz", "/collections"):
        try:
            response = httpx.get(url + path, headers=headers, timeout=10.0)
        except Exception as exc:
            print(f"  {path:14s} -> EXC {exc.__class__.__name__}: {exc}")
            ok = False
            continue

        content_type = response.headers.get("content-type", "?")
        body = response.text.strip()[:160]
        print(
            f"  {path:14s} -> HTTP {response.status_code}  "
            f"ct={content_type}  body={body!r}"
        )

        if response.status_code != 200:
            ok = False
            continue

        # Qdrant's health endpoint returns plain text; collections returns JSON.
        if path == "/collections":
            try:
                payload = response.json()
            except ValueError:
                ok = False
            else:
                if payload.get("status") != "ok" or "result" not in payload:
                    ok = False

    if ok:
        print(f"\n✓ {args.target.title()} Qdrant is healthy — safe to run the notebook.")
        return 0

    if args.target == "local":
        print(
            "\n✗ Local Qdrant is not healthy.\n"
            "  Run ./start_qdrant.sh, then inspect: docker logs qdrant"
        )
    else:
        print(
            "\n✗ Qdrant Cloud did not respond like a live cluster.\n"
            "  Check that the cluster is active and QDRANT_URL/QDRANT_API_KEY\n"
            "  belong to the same cluster: https://cloud.qdrant.io/"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
