import argparse
import csv
import json
import os
from typing import Dict

from ._runner_common import build_arg_parser, run_benchmark
from .format_outputs import write_metrics_json
from .scoring import build_metrics_schema

TARGET = {
    "bamboogle": 58.4,
    "2wiki": 60.0,
    "hotpotqa": 51.3,
    "musique": 19.2,
    "avg_search_intensive": 47.2,
    "gaia": 17.2,
}


def _metric_or_zero(m: Dict) -> float:
    return float(m.get("accuracy", 0.0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--engine_type", type=str, default="openai_compat")
    p.add_argument("--base_url", type=str, default=os.getenv("OPENAI_COMPAT_BASE_URL", ""))
    p.add_argument("--api_key", type=str, default=os.getenv("OPENAI_COMPAT_API_KEY", ""))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_examples", type=int, default=100)
    p.add_argument("--upstream_repo", type=str, default="")
    p.add_argument("--wiki_jsonl_path", type=str, default="")
    p.add_argument("--bamboogle_path", type=str, default="")
    p.add_argument("--wiki2_path", type=str, default="")
    p.add_argument("--hotpotqa_path", type=str, default="")
    p.add_argument("--musique_path", type=str, default="")
    p.add_argument("--gaia_path", type=str, default="")
    p.add_argument("--output_dir", type=str, default="final_project_part2/outputs/table1_row_repro")
    args = p.parse_args()

    bench_paths = {
        "bamboogle": args.bamboogle_path,
        "2wiki": args.wiki2_path,
        "hotpotqa": args.hotpotqa_path,
        "musique": args.musique_path,
        "gaia": args.gaia_path,
    }

    benchmark_metrics: Dict[str, Dict] = {}
    for bench, ds_path in bench_paths.items():
        if not ds_path:
            benchmark_metrics[bench] = {"accuracy": 0.0, "status": "missing_dataset_path"}
            continue
        if not args.base_url or not args.api_key:
            benchmark_metrics[bench] = {"accuracy": 0.0, "status": "missing_endpoint_config"}
            continue

        bench_args = argparse.Namespace(
            dataset_path=ds_path,
            n_examples=args.n_examples,
            seed=args.seed,
            upstream_repo=args.upstream_repo,
            trace_out=os.path.join(args.output_dir, f"trace_{bench}.jsonl"),
            engine_type=args.engine_type,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_sec=60,
            max_retries=3,
            log_dir=os.path.join(args.output_dir, "engine_logs"),
            wiki_jsonl_path=args.wiki_jsonl_path,
            max_turns=4,
        )
        try:
            metric = run_benchmark(bench, bench_args)
            if "accuracy" not in metric:
                metric = {"accuracy": 0.0, "status": "upstream_invoked"}
            benchmark_metrics[bench] = metric
        except Exception as e:
            benchmark_metrics[bench] = {"accuracy": 0.0, "status": f"error: {e}"}

    metrics = build_metrics_schema(args.model, args.engine_type, args.seed, benchmark_metrics)

    model_dir = os.path.join(args.output_dir, args.model.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    out_json = os.path.join(model_dir, "metrics.json")
    write_metrics_json(out_json, metrics)

    comparison_path = os.path.join(model_dir, "target_comparison.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "ours", "target", "delta"])
        for key in ["bamboogle", "2wiki", "hotpotqa", "musique"]:
            ours = _metric_or_zero(benchmark_metrics.get(key, {}))
            tgt = TARGET[key]
            w.writerow([key, f"{ours:.3f}", f"{tgt:.3f}", f"{ours - tgt:.3f}"])
        ours_avg = metrics["avg_search_intensive"]
        w.writerow(["avg_search_intensive", f"{ours_avg:.3f}", f"{TARGET['avg_search_intensive']:.3f}", f"{ours_avg - TARGET['avg_search_intensive']:.3f}"])
        ours_gaia = metrics["gaia"]
        w.writerow(["gaia", f"{ours_gaia:.3f}", f"{TARGET['gaia']:.3f}", f"{ours_gaia - TARGET['gaia']:.3f}"])

    print(json.dumps(metrics, indent=2))
    print(f"Wrote: {out_json}")
    print(f"Wrote: {comparison_path}")


if __name__ == "__main__":
    main()
