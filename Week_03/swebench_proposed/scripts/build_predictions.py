#!/usr/bin/env python3
"""Build official harness predictions JSONL from swe_search outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out)
    source = run_dir / "search_outputs.json"
    rows = json.loads(source.read_text(encoding="utf-8"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "instance_id": row["instance_id"],
                "model_name_or_path": row.get("model_name_or_path", "unknown"),
                "model_patch": row.get("model_patch", ""),
            }
            handle.write(json.dumps(payload) + "\n")

    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
