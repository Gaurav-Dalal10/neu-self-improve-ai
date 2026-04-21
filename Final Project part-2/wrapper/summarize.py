import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from .env_setup import resolve_agentflow_dir

TARGET = {
    "bamboogle": 58.4,
    "2wiki": 60.0,
    "hotpotqa": 51.3,
    "musique": 19.2,
    "avg": 47.2,
    "gaia": 17.2,
}

TASKS = ["bamboogle", "2wiki", "hotpotqa", "musique", "gaia"]
SEARCH_TASKS = ["bamboogle", "2wiki", "hotpotqa", "musique"]


def find_final_score_file(agentflow_dir: Path, task: str, model_label: str) -> Optional[Path]:
    exact = agentflow_dir / "test" / task / "results" / model_label / "final_scores_direct_output.json"
    if exact.exists():
        return exact

    root = agentflow_dir / "test" / task / "results"
    if not root.exists():
        return None

    candidates = sorted(root.glob("**/final_scores_direct_output.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if model_label in str(c):
            return c
    return candidates[0] if candidates else None


def parse_accuracy_from_final_scores(path: Path) -> float:
    obj = json.loads(path.read_text(encoding="utf-8"))

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = k.lower()
                if lk in {"accuracy", "acc", "score"} and isinstance(v, (int, float)):
                    yield float(v)
                else:
                    yield from walk(v)
        elif isinstance(x, list):
            for it in x:
                yield from walk(it)

    vals = list(walk(obj))
    if not vals:
        raise ValueError(f"No numeric accuracy/acc/score found in {path}")
    # Choose first deterministic extracted metric.
    return vals[0]


def aggregate_metrics(scores: Dict[str, float]) -> Dict[str, float]:
    vals = [scores[t] for t in SEARCH_TASKS if t in scores]
    avg = sum(vals) / len(vals) if vals else 0.0
    out = dict(scores)
    out["avg"] = avg
    return out


def write_outputs(out_dir: Path, metrics: Dict[str, float], sources: Dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metrics": metrics,
        "targets": TARGET,
        "deltas": {k: metrics.get(k, 0.0) - TARGET.get(k, 0.0) for k in TARGET.keys()},
        "sources": sources,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "ours", "target", "delta", "source_file"])
        for k in ["bamboogle", "2wiki", "hotpotqa", "musique", "avg", "gaia"]:
            w.writerow([
                k,
                f"{metrics.get(k, 0.0):.6f}",
                f"{TARGET.get(k, 0.0):.6f}",
                f"{metrics.get(k, 0.0) - TARGET.get(k, 0.0):.6f}",
                sources.get(k, ""),
            ])


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize upstream AgentFlow benchmark outputs")
    p.add_argument("--model_label", required=True)
    p.add_argument("--agentflow_dir", default="")
    p.add_argument("--output_dir", default="final_project_part2/outputs/row_repro")
    args = p.parse_args()

    af_dir = resolve_agentflow_dir(args.agentflow_dir or None)

    scores: Dict[str, float] = {}
    sources: Dict[str, str] = {}
    for t in TASKS:
        path = find_final_score_file(af_dir, t, args.model_label)
        if path is None:
            continue
        try:
            scores[t] = parse_accuracy_from_final_scores(path)
            sources[t] = str(path)
        except Exception:
            continue

    metrics = aggregate_metrics(scores)
    if "gaia" not in metrics:
        metrics["gaia"] = scores.get("gaia", 0.0)

    out_dir = Path(args.output_dir) / args.model_label
    write_outputs(out_dir, metrics, sources)
    print(json.dumps({"metrics": metrics, "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        raise SystemExit(2)
