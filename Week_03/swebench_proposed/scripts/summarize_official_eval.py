#!/usr/bin/env python3
"""Summarize official SWE-bench evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_json_candidates(eval_dir: Path) -> list[Path]:
    return sorted(eval_dir.rglob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out = Path(args.out)

    resolved = 0
    total = 0
    details = []
    aggregate_source = None

    for path in _find_json_candidates(eval_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(payload, dict):
            # Official SWE-bench aggregate report (schema_version=2)
            if "submitted_instances" in payload and "resolved_instances" in payload:
                aggregate_source = str(path)
                total = int(payload.get("submitted_instances") or 0)
                resolved = int(payload.get("resolved_instances") or 0)
                details = [
                    {"instance_id": iid, "resolved": 1}
                    for iid in payload.get("resolved_ids", [])
                ] + [
                    {"instance_id": iid, "resolved": 0}
                    for iid in payload.get("unresolved_ids", [])
                ]
                # Prefer official aggregate when present.
                continue

            for key in ("instance_id", "resolved", "status", "report"):
                if key in payload:
                    break
            else:
                continue

            if "instance_id" in payload:
                # Only use per-instance rows when official aggregate is absent.
                if aggregate_source:
                    continue
                total += 1
                val = payload.get("resolved")
                status = str(payload.get("status", "")).lower()
                is_resolved = bool(val) or status == "resolved"
                resolved += int(is_resolved)
                details.append({
                    "instance_id": payload.get("instance_id"),
                    "resolved": int(is_resolved),
                    "source": str(path),
                })

    summary = {
        "eval_dir": str(eval_dir),
        "source": aggregate_source or "per-instance-json",
        "total": total,
        "resolved": resolved,
        "resolved_rate": round((resolved / total), 4) if total else 0.0,
        "details": details,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
