from final_project_part2.agentflow_wrapper.eval.scoring import build_metrics_schema


def test_scoring_schema():
    metrics = build_metrics_schema(
        model="qwen",
        engine="openai_compat",
        seed=1,
        benchmark_metrics={
            "bamboogle": {"accuracy": 10.0},
            "2wiki": {"accuracy": 20.0},
            "hotpotqa": {"accuracy": 30.0},
            "musique": {"accuracy": 40.0},
            "gaia": {"accuracy": 5.0},
        },
    )
    assert "avg_search_intensive" in metrics
    assert abs(metrics["avg_search_intensive"] - 25.0) < 1e-9
    assert metrics["gaia"] == 5.0
