import argparse
import csv
import json
import os
import subprocess
from typing import Any, Dict, List

from ..agent.agentflow_core import AgentFlowCore
from ..engine_clients import make_client
from .scoring import accuracy
from .subsets import sample_examples


def _load_rows(dataset_path: str) -> List[Dict[str, Any]]:
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

    rows: List[Dict[str, Any]] = []
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    elif dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            rows = obj if isinstance(obj, list) else obj.get("data", [])
    else:
        raise ValueError("Dataset must be .jsonl or .json")
    return rows


def _call_upstream_if_available(upstream_repo: str, benchmark_name: str) -> bool:
    if not upstream_repo:
        return False
    run_sh = os.path.join(upstream_repo, "test", benchmark_name, "run.sh")
    if not os.path.exists(run_sh):
        return False
    subprocess.run(["bash", run_sh], check=False)
    return True


def run_benchmark(name: str, args: argparse.Namespace) -> Dict[str, Any]:
    if _call_upstream_if_available(args.upstream_repo, name):
        return {"benchmark": name, "mode": "upstream", "note": "Executed upstream run.sh"}

    rows = _load_rows(args.dataset_path)
    rows = sample_examples(rows, args.n_examples, args.seed)

    client = make_client(
        engine_type=args.engine_type,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        log_dir=args.log_dir,
    )
    core = AgentFlowCore(client=client, wiki_jsonl_path=args.wiki_jsonl_path)

    correct = 0
    traces = []
    for i, row in enumerate(rows):
        question = row.get("question") or row.get("query") or ""
        gold = str(row.get("answer", "")).strip().lower()
        out = core.run(question=question, max_turns=args.max_turns)
        pred = str(out.get("final_answer", "")).strip().lower()
        ok = int(bool(gold) and gold in pred)
        correct += ok
        traces.append({
            "idx": i,
            "question": question,
            "gold": gold,
            "pred": pred,
            "correct": ok,
            "trace": out.get("trace", []),
        })

    metric = {
        "accuracy": accuracy(correct, len(rows)),
        "correct": correct,
        "total": len(rows),
        "mode": "wrapper",
    }

    if args.trace_out:
        os.makedirs(os.path.dirname(args.trace_out), exist_ok=True)
        with open(args.trace_out, "w", encoding="utf-8") as f:
            for row in traces:
                f.write(json.dumps(row) + "\n")

    return metric


def build_arg_parser(default_benchmark: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Run benchmark: {default_benchmark}")
    p.add_argument("--dataset_path", type=str, default="")
    p.add_argument("--n_examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--upstream_repo", type=str, default="")
    p.add_argument("--trace_out", type=str, default="")

    p.add_argument("--engine_type", type=str, default="openai_compat")
    p.add_argument("--base_url", type=str, default=os.getenv("OPENAI_COMPAT_BASE_URL", ""))
    p.add_argument("--api_key", type=str, default=os.getenv("OPENAI_COMPAT_API_KEY", ""))
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--timeout_sec", type=int, default=60)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--log_dir", type=str, default="")

    p.add_argument("--wiki_jsonl_path", type=str, default="")
    p.add_argument("--max_turns", type=int, default=4)
    return p
