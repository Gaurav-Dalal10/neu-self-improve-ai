"""Analysis script for test-based SWE-bench-style evaluation results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from database import DatabaseManager


def _fetch_latest_eval_rows(db: DatabaseManager) -> List[dict]:
    cursor = db.conn.cursor()
    cursor.execute(
        """
        SELECT
            eval_run_name,
            instance_id,
            repo,
            base_commit,
            patch_path,
            tests_command,
            tests_exit_code,
            resolved,
            patch_generated,
            runtime_seconds,
            error_type,
            logs_path,
            created_at
        FROM evaluation_runs
        ORDER BY created_at ASC
        """
    )
    return [dict(row) for row in cursor.fetchall()]


def _aggregate_unique(rows: List[dict]) -> List[dict]:
    deduped: Dict[str, dict] = {}
    for row in rows:
        deduped[row["instance_id"]] = row
    return list(deduped.values())


def analyze_results(db_path: str = "swe_bench.db"):
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return

    with DatabaseManager(db_path) as db:
        rows = _fetch_latest_eval_rows(db)
        if not rows:
            _analyze_legacy_results(db)
            return

    unique_rows = _aggregate_unique(rows)

    total_runs = len(rows)
    resolved_runs = sum(int(r["resolved"]) for r in rows)
    patch_generated_runs = sum(int(r["patch_generated"]) for r in rows)

    total_unique = len(unique_rows)
    resolved_unique = sum(int(r["resolved"]) for r in unique_rows)
    patch_generated_unique = sum(int(r["patch_generated"]) for r in unique_rows)

    avg_runtime = sum(float(r.get("runtime_seconds") or 0.0) for r in rows) / total_runs

    print("=" * 72)
    print("SWE-bench Baseline: Test-Based Evaluation Analysis")
    print("=" * 72)
    print("\nPer-run metrics")
    print("-" * 72)
    print(f"Total runs: {total_runs}")
    print(f"Resolved runs: {resolved_runs}")
    print(f"Resolved rate (runs): {resolved_runs / total_runs:.2%}")
    print(f"Patch-generated runs: {patch_generated_runs}")
    print(f"Patch-generated rate (runs): {patch_generated_runs / total_runs:.2%}")
    print(f"Average runtime seconds: {avg_runtime:.2f}")

    print("\nPer-unique-instance metrics (deduplicated by latest run)")
    print("-" * 72)
    print(f"Unique instances: {total_unique}")
    print(f"Resolved unique instances: {resolved_unique}")
    print(f"Resolved rate (pass@1-style): {resolved_unique / total_unique:.2%}")
    print(f"Patch-generated unique instances: {patch_generated_unique}")
    print(
        f"Patch-generated rate (unique): {patch_generated_unique / total_unique:.2%}"
    )


def _analyze_legacy_results(db: DatabaseManager):
    """Fallback analysis for older patch-generated evaluation rows."""
    print("[info] No rows in evaluation_runs. Falling back to legacy evaluation_results.")
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM evaluation_results")
    total = cursor.fetchone()[0]
    if total == 0:
        print("No evaluation data found.")
        return

    cursor.execute("SELECT SUM(resolved) FROM evaluation_results")
    resolved = cursor.fetchone()[0] or 0
    print("=" * 72)
    print("SWE-bench Baseline: Legacy Analysis (Patch-Generated Metric)")
    print("=" * 72)
    print(f"Total evaluations: {total}")
    print(f"Resolved (legacy): {resolved}")
    print(f"Legacy resolve rate: {resolved / total:.2%}")
    print("Use `python -m eval.run ...` for test-based resolved/pass@1.")


def export_results_csv(db_path: str = "swe_bench.db", output_file: str = "results.csv"):
    with DatabaseManager(db_path) as db:
        rows = _fetch_latest_eval_rows(db)

    if not rows:
        with DatabaseManager(db_path) as db:
            cursor = db.conn.cursor()
            cursor.execute(
                """
                SELECT
                    e.instance_id,
                    i.repo,
                    i.base_commit,
                    '' AS patch_path,
                    '' AS tests_command,
                    NULL AS tests_exit_code,
                    e.resolved,
                    NULL AS runtime_seconds,
                    'legacy_metric' AS error_type,
                    '' AS logs_path,
                    e.resolved AS patch_generated,
                    '' AS eval_run_name,
                    e.eval_timestamp AS created_at
                FROM evaluation_results e
                JOIN swe_bench_instances i ON i.instance_id = e.instance_id
                ORDER BY e.eval_timestamp ASC
                """
            )
            rows = [dict(r) for r in cursor.fetchall()]
        if not rows:
            print("No evaluation rows to export.")
            return

    headers = [
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
        "eval_run_name",
        "created_at",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})

    print(f"Exported {len(rows)} rows to {output_file}")


def show_trajectory(instance_id: str, db_path: str = "swe_bench.db"):
    with DatabaseManager(db_path) as db:
        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT trajectory_id, model_name, total_actions, success, final_patch
            FROM agent_trajectories
            WHERE instance_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (instance_id,),
        )
        traj = cursor.fetchone()

        if not traj:
            print(f"No trajectory found for instance: {instance_id}")
            return

        print("=" * 72)
        print(f"Trajectory for: {instance_id}")
        print("=" * 72)
        print(f"Trajectory ID: {traj['trajectory_id']}")
        print(f"Model: {traj['model_name']}")
        print(f"Total actions: {traj['total_actions']}")
        print(f"Patch generated: {bool(traj['success'])}")
        print(f"Patch length: {len(traj['final_patch'] or '')}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--export":
            export_results_csv()
        elif sys.argv[1] == "--trajectory":
            if len(sys.argv) > 2:
                show_trajectory(sys.argv[2])
            else:
                print("Usage: python analyze.py --trajectory <instance_id>")
        else:
            print("Usage:")
            print("  python analyze.py")
            print("  python analyze.py --export")
            print("  python analyze.py --trajectory <id>")
    else:
        analyze_results()
