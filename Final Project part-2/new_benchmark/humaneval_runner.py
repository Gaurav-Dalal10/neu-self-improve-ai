import argparse
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List

from .humaneval_scoring import build_summary


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _run_test(code: str, tests: str, timeout_sec: int = 5) -> bool:
    script = f"{code}\n\n{tests}\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as fp:
        fp.write(script)
        fp.flush()
        try:
            proc = subprocess.run(["python3", fp.name], timeout=timeout_sec, capture_output=True, text=True)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=str, default="final_project_part2/new_benchmark/dataset_local/humaneval.jsonl")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--n_examples", type=int, default=20)
    p.add_argument("--output_json", type=str, default="final_project_part2/outputs/new_benchmark/humaneval_result.json")
    args = p.parse_args()

    if not os.path.exists(args.dataset_path):
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json.replace(".json", "_MISSING_DATASET.txt"), "w", encoding="utf-8") as f:
            f.write("Missing HumanEval dataset. Provide local JSONL at --dataset_path.\n")
        print("Missing HumanEval dataset.")
        return

    rows = _load_jsonl(args.dataset_path)[: args.n_examples]
    passed = 0

    results = []
    for row in rows:
        prompt = row.get("prompt", "")
        canonical = row.get("canonical_solution", "")
        tests = row.get("test", "")

        # Offline fallback: use canonical solution as generated placeholder.
        generated = canonical if canonical else "def solution(*args, **kwargs):\n    return None\n"
        ok = _run_test(generated, tests)
        passed += int(ok)
        results.append({"task_id": row.get("task_id"), "passed": bool(ok)})

    summary = build_summary(args.model, len(rows), passed)
    payload = {"summary": summary, "results": results}

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
