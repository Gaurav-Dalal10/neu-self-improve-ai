import argparse
import csv
import os
from typing import Dict, List
import yaml

from ._runner_common import run_benchmark

def _row_for_missing(label: str, model: str, status: str) -> Dict[str, str]:
    return {
        "label": label,
        "model": model,
        "bamboogle": "",
        "2wiki": "",
        "hotpotqa": "",
        "musique": "",
        "avg_search": "",
        "gaia": "",
        "status": status,
    }

def _load_sweep_models(config_path: str) -> List[Dict[str, str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("sweep_models", [])
    if not models:
        raise ValueError(f"No sweep_models defined in {config_path}")
    return models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", type=str, default=os.getenv("OPENAI_COMPAT_BASE_URL", ""))
    p.add_argument("--api_key", type=str, default=os.getenv("OPENAI_COMPAT_API_KEY", ""))
    p.add_argument("--engine_type", type=str, default="openai_compat")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_examples", type=int, default=100)
    p.add_argument("--bamboogle_path", type=str, default="")
    p.add_argument("--wiki2_path", type=str, default="")
    p.add_argument("--hotpotqa_path", type=str, default="")
    p.add_argument("--musique_path", type=str, default="")
    p.add_argument("--gaia_path", type=str, default="")
    p.add_argument("--models_config", type=str, default="final_project_part2/configs/qwen_models.yaml")
    p.add_argument("--output_csv", type=str, default="final_project_part2/outputs/qwen35_no_training/summary.csv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    sweep_models = _load_sweep_models(args.models_config)

    if not args.base_url or not args.api_key:
        rows = [_row_for_missing(m.get("label", "unknown"), m.get("model", ""), "missing_endpoint_config") for m in sweep_models]
    else:
        bench_paths = {
            "bamboogle": args.bamboogle_path,
            "2wiki": args.wiki2_path,
            "hotpotqa": args.hotpotqa_path,
            "musique": args.musique_path,
            "gaia": args.gaia_path,
        }
        rows = []
        for entry in sweep_models:
            label = entry.get("label", "unknown")
            model = entry.get("model", "")
            if not all(bench_paths.values()):
                rows.append(_row_for_missing(label, model, "missing_dataset_path"))
                continue

            metrics = {}
            status = "ok"
            for bench, path in bench_paths.items():
                bench_args = argparse.Namespace(
                    dataset_path=path,
                    n_examples=args.n_examples,
                    seed=args.seed,
                    upstream_repo="",
                    trace_out="",
                    engine_type=args.engine_type,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=model,
                    timeout_sec=60,
                    max_retries=3,
                    log_dir="",
                    wiki_jsonl_path="",
                    max_turns=4,
                )
                try:
                    m = run_benchmark(bench, bench_args)
                    metrics[bench] = float(m.get("accuracy", 0.0))
                except Exception:
                    status = "partial_error"
                    metrics[bench] = 0.0

            avg = (metrics.get("bamboogle", 0.0) + metrics.get("2wiki", 0.0) + metrics.get("hotpotqa", 0.0) + metrics.get("musique", 0.0)) / 4.0
            rows.append(
                {
                    "label": label,
                    "model": model,
                    "bamboogle": f"{metrics.get('bamboogle', 0.0):.3f}",
                    "2wiki": f"{metrics.get('2wiki', 0.0):.3f}",
                    "hotpotqa": f"{metrics.get('hotpotqa', 0.0):.3f}",
                    "musique": f"{metrics.get('musique', 0.0):.3f}",
                    "avg_search": f"{avg:.3f}",
                    "gaia": f"{metrics.get('gaia', 0.0):.3f}",
                    "status": status,
                }
            )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["label", "model", "bamboogle", "2wiki", "hotpotqa", "musique", "avg_search", "gaia", "status"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
