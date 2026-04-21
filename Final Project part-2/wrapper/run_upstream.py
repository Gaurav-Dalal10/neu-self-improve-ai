import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

from .env_setup import canonicalize_openai_env, ensure_agentflow_env, resolve_agentflow_dir


def _has_fatal_errors(log_path: Path) -> bool:
    if not log_path.exists():
        return True
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = [
        "Traceback (most recent call last):",
        "ModuleNotFoundError:",
        "ImportError:",
        "ValueError:",
        "Exception:",
    ]
    return any(p in txt for p in patterns)


def run_task(agentflow_dir: Path, task: str, model_label: str, output_root: Path, strict: bool = True) -> Dict:
    task_dir = agentflow_dir / "test" / task
    run_sh = task_dir / "run.sh"
    if not run_sh.exists():
        raise FileNotFoundError(f"Missing benchmark script: {run_sh}")

    canonicalize_openai_env()
    ensure_agentflow_env(agentflow_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"{task}_run.log"

    env = os.environ.copy()
    env["MODEL_LABEL"] = model_label

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            ["bash", str(run_sh)],
            cwd=str(task_dir),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )

    result = {
        "task": task,
        "run_script": str(run_sh),
        "log": str(log_path),
        "returncode": proc.returncode,
    }
    if _has_fatal_errors(log_path):
        result["fatal_errors_in_log"] = True
        if strict:
            raise RuntimeError(
                f"Task {task} produced fatal errors. Check log: {log_path}"
            )
    else:
        result["fatal_errors_in_log"] = False
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Run upstream AgentFlow benchmark task script")
    p.add_argument("--task", required=True, choices=["bamboogle", "2wiki", "hotpotqa", "musique", "gaia"])
    p.add_argument("--model_label", required=True)
    p.add_argument("--agentflow_dir", default="")
    p.add_argument("--output_dir", default="final_project_part2/outputs/row_repro")
    p.add_argument("--strict", action="store_true", default=False)
    args = p.parse_args()

    af_dir = resolve_agentflow_dir(args.agentflow_dir or None)
    out_dir = Path(args.output_dir) / args.model_label
    result = run_task(af_dir, args.task, args.model_label, out_dir, strict=args.strict)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        raise SystemExit(2)
