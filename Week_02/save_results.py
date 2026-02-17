"""Save test-based evaluation summary to a text file."""

from __future__ import annotations

from pathlib import Path

from database import DatabaseManager


def save_results_to_file(output_file: str = "results_summary.txt", db_path: str = "swe_bench.db"):
    with DatabaseManager(db_path) as db:
        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT instance_id, resolved, patch_generated, runtime_seconds, error_type, created_at
            FROM evaluation_runs
            ORDER BY created_at ASC
            """
        )
        rows = [dict(row) for row in cursor.fetchall()]
        if not rows:
            cursor.execute(
                """
                SELECT instance_id, resolved, resolved AS patch_generated,
                       NULL AS runtime_seconds, 'legacy_metric' AS error_type,
                       eval_timestamp AS created_at
                FROM evaluation_results
                ORDER BY eval_timestamp ASC
                """
            )
            rows = [dict(row) for row in cursor.fetchall()]

    if not rows:
        Path(output_file).write_text(
            "No evaluations found. Run python agent.py and/or python -m eval.run first.\n",
            encoding="utf-8",
        )
        print(f"Results saved to {output_file}")
        return

    total = len(rows)
    resolved = sum(int(r["resolved"]) for r in rows)
    patch_generated = sum(int(r["patch_generated"]) for r in rows)
    avg_runtime = sum(float(r.get("runtime_seconds") or 0.0) for r in rows) / total

    lines = [
        "=" * 72,
        "SWE-bench Baseline: Test-Based Evaluation Summary",
        "=" * 72,
        f"Total runs: {total}",
        f"Resolved runs: {resolved}",
        f"Resolved rate: {resolved / total:.2%}",
        f"Patch-generated runs: {patch_generated}",
        f"Patch-generated rate: {patch_generated / total:.2%}",
        f"Average runtime seconds: {avg_runtime:.2f}",
        "",
        "Latest 10 rows:",
        "-" * 72,
    ]

    for row in rows[-10:]:
        lines.append(
            f"{row['instance_id']} | resolved={row['resolved']} | "
            f"patch_generated={row['patch_generated']} | "
            f"runtime={row['runtime_seconds']} | error={row['error_type']}"
        )

    Path(output_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    save_results_to_file()
