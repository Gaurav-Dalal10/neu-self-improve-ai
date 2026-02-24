"""Wrapper for official SWE-bench harness execution."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_official_eval(
    dataset_name: str,
    predictions_path: Path,
    run_id: str,
    max_workers: int = 1,
) -> int:
    cmd = [
        "python3",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    completed = subprocess.run(cmd)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official SWE-bench harness evaluation")
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    rc = run_official_eval(
        dataset_name=args.dataset_name,
        predictions_path=Path(args.predictions_path),
        run_id=args.run_id,
        max_workers=args.max_workers,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
