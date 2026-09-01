import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config_openrouter_qdrant.yaml"
OUTPUT_PATH = Path(__file__).parent / "missing_metadata_report.md"
CONTENT_SNIPPET_LEN = 300


def resolve_env(value: str) -> str:
    match = re.match(r"\$\{(.+)\}", value)
    if match:
        return os.getenv(match.group(1), "")
    return value


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

vs = config["vectorstore"]
client = QdrantClient(
    url=resolve_env(vs["url"]),
    api_key=resolve_env(vs.get("api_key", "")),
)
collection = vs["collection_name"]

total = 0
missing_records: list[dict] = []
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
        total += 1
        payload = point.payload or {}

        meta = payload.get("metadata")
        if meta is None:
            if "page_content" in payload:
                meta = {k: v for k, v in payload.items() if k != "page_content"}
            else:
                meta = payload

        null_fields = [k for k, v in meta.items() if v is None]
        if null_fields:
            page_content = payload.get("page_content", "")
            missing_records.append({
                "point_id": point.id,
                "source": meta.get("source", "unknown"),
                "file_name": meta.get("file_name", "unknown"),
                "chunk_id": meta.get("chunk_id", "unknown"),
                "chunk_index": meta.get("chunk_index", "unknown"),
                "null_fields": null_fields,
                "page_content": page_content[:CONTENT_SNIPPET_LEN],
                "all_metadata": meta,
            })

    if offset is None:
        break

missing_records.sort(key=lambda r: (r["file_name"], r["chunk_index"]))

grouped: dict[str, list[dict]] = defaultdict(list)
for rec in missing_records:
    grouped[rec["file_name"]].append(rec)

lines: list[str] = []
lines.append("# Missing Metadata Report")
lines.append("")
lines.append(f"- **Collection:** `{collection}`")
lines.append(f"- **Cluster:** AGENTIC_AI")
lines.append(f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
lines.append(f"- **Total records:** {total}")
lines.append(f"- **Records with missing metadata:** {len(missing_records)}")
lines.append(f"- **Affected files:** {len(grouped)}")
lines.append("")
lines.append("---")
lines.append("")

for file_name, records in grouped.items():
    lines.append(f"## {file_name}")
    lines.append("")
    lines.append(f"**Missing records in this file:** {len(records)}")
    lines.append("")

    for rec in records:
        lines.append(f"### Point ID: `{rec['point_id']}`")
        lines.append("")
        lines.append(f"- **Source:** `{rec['source']}`")
        lines.append(f"- **Chunk ID:** `{rec['chunk_id']}`")
        lines.append(f"- **Chunk Index:** {rec['chunk_index']}")
        lines.append(f"- **Null fields ({len(rec['null_fields'])}):** {', '.join(f'`{f}`' for f in rec['null_fields'])}")
        lines.append("")

        lines.append("<details>")
        lines.append("<summary>Page Content Preview</summary>")
        lines.append("")
        lines.append("```")
        lines.append(rec["page_content"] if rec["page_content"] else "(empty)")
        lines.append("```")
        lines.append("</details>")
        lines.append("")

        lines.append("<details>")
        lines.append("<summary>Full Metadata</summary>")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        for k, v in rec["all_metadata"].items():
            val = f"`{escape_md(str(v))}`" if v is not None else "**null**"
            lines.append(f"| `{k}` | {val} |")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    lines.append("")

OUTPUT_PATH.write_text("\n".join(lines))

print(f"Total records: {total}")
print(f"Records with missing metadata: {len(missing_records)}")
print(f"Report written to: {OUTPUT_PATH}")
