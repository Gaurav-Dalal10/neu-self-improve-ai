from typing import Dict


def accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(correct) / float(total)


def build_metrics_schema(model: str, engine: str, seed: int, benchmark_metrics: Dict[str, Dict[str, float]]) -> Dict:
    search_keys = ["bamboogle", "2wiki", "hotpotqa", "musique"]
    search_vals = [benchmark_metrics.get(k, {}).get("accuracy", 0.0) for k in search_keys]
    avg = sum(search_vals) / len(search_vals) if search_vals else 0.0
    gaia = benchmark_metrics.get("gaia", {}).get("accuracy", 0.0)

    return {
        "model": model,
        "engine": engine,
        "seed": seed,
        "benchmarks": benchmark_metrics,
        "avg_search_intensive": avg,
        "gaia": gaia,
    }
