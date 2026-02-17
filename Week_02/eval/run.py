"""CLI entrypoint for test-based SWE-bench-style evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from database import DatabaseManager
from eval.harness import EvaluationConfig, EvaluationRecord, LiteHarness, aggregate_records


def _load_rows_for_evaluation(
    db: DatabaseManager,
    split: str,
    limit: Optional[int],
    instance_id: Optional[str],
    model_name: Optional[str],
) -> List[Dict[str, Any]]:
    cursor = db.conn.cursor()

    where_parts = []
    args: List[Any] = []

    if instance_id:
        where_parts.append("i.instance_id = ?")
        args.append(instance_id)

    if model_name:
        where_parts.append("t.model_name = ?")
        args.append(model_name)

    # Existing DB does not store split metadata. Keep argument for compatibility.
    if split and split != "test":
        print(f"[warn] split '{split}' requested, but DB has no split column. Using available rows.")

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    query = f"""
        SELECT
            i.instance_id,
            i.repo,
            i.base_commit,
            i.FAIL_TO_PASS,
            t.trajectory_id,
            t.model_name,
            t.timestamp,
            t.final_patch
        FROM agent_trajectories t
        JOIN swe_bench_instances i ON i.instance_id = t.instance_id
        {where_sql}
        ORDER BY t.timestamp ASC
    """

    if limit is not None:
        query += " LIMIT ?"
        args.append(limit)

    cursor.execute(query, args)
    rows = [dict(row) for row in cursor.fetchall()]

    for row in rows:
        fail_to_pass = row.get("FAIL_TO_PASS")
        if fail_to_pass:
            try:
                row["FAIL_TO_PASS"] = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                row["FAIL_TO_PASS"] = []
        else:
            row["FAIL_TO_PASS"] = []

    return rows


def _write_per_run_csv(records: List[EvaluationRecord], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instance_id",
                "repo",
                "base_commit",
                "patch_path",
                "tests_command",
                "tests_exit_code",
                "resolved",
                "runtime_seconds",
                "error_type",
                "logs_path",
                "patch_generated",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_summary_json(
    records: List[EvaluationRecord],
    aggregates: Dict[str, Any],
    output_path: Path,
):
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "aggregates": aggregates,
        "records": [asdict(record) for record in records],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _persist_records(db: DatabaseManager, records: List[EvaluationRecord], eval_run_name: str):
    for record in records:
        db.save_eval_run(
            eval_run_name=eval_run_name,
            instance_id=record.instance_id,
            repo=record.repo,
            base_commit=record.base_commit,
            patch_path=record.patch_path,
            tests_command=record.tests_command,
            tests_exit_code=record.tests_exit_code,
            resolved=record.resolved,
            patch_generated=record.patch_generated,
            runtime_seconds=record.runtime_seconds,
            error_type=record.error_type,
            logs_path=record.logs_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Run test-based evaluation for SWE-bench baseline patches")
    parser.add_argument("--db", default="swe_bench.db", help="Path to SQLite DB")
    parser.add_argument("--split", default="test", help="Dataset split label (compatibility arg)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trajectories")
    parser.add_argument("--instance", default=None, help="Evaluate a single instance_id")
    parser.add_argument("--model", default=None, help="Filter trajectories by model name")
    parser.add_argument("--tests_command", default=None, help="Override test command for all instances")
    parser.add_argument("--timeout", type=int, default=900, help="Test timeout per instance in seconds")
    parser.add_argument("--max_log_bytes", type=int, default=200000, help="Max stdout/stderr bytes per log")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: runs/<YYYYMMDD_HHMMSS>",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EvaluationConfig(
        timeout_seconds=args.timeout,
        max_log_bytes=args.max_log_bytes,
    )
    harness = LiteHarness(config=config)

    with DatabaseManager(args.db) as db:
        rows = _load_rows_for_evaluation(
            db=db,
            split=args.split,
            limit=args.limit,
            instance_id=args.instance,
            model_name=args.model,
        )

        if not rows:
            print("No trajectories found to evaluate.")
            return

        eval_run_name = output_dir.name
        records: List[EvaluationRecord] = []

        for idx, row in enumerate(rows, 1):
            print(f"[{idx}/{len(rows)}] Evaluating {row['instance_id']} (trajectory {row['trajectory_id']})")
            instance_data = {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "FAIL_TO_PASS": row.get("FAIL_TO_PASS", []),
            }

            record = harness.evaluate_instance(
                instance=instance_data,
                patch_text=row.get("final_patch") or "",
                output_dir=output_dir,
                tests_command_override=args.tests_command,
            )
            records.append(record)

        aggregates = aggregate_records(records)
        _persist_records(db, records, eval_run_name=eval_run_name)

    per_run_csv = output_dir / "evaluation_results.csv"
    summary_json = output_dir / "summary.json"
    _write_per_run_csv(records, per_run_csv)
    _write_summary_json(records, aggregates, summary_json)

    print("\nEvaluation complete")
    print(f"Per-run CSV: {per_run_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Resolved rate (runs): {aggregates['resolved_rate_runs']:.2%}")
    print(f"Resolved rate (unique instances): {aggregates['resolved_rate_unique_instances']:.2%}")
    print(f"Patch-generated rate (runs): {aggregates['patch_generated_rate_runs']:.2%}")


if __name__ == "__main__":
    main()
