from final_project_part2.wrapper.summarize import aggregate_metrics


def test_summary_aggregation_avg():
    m = aggregate_metrics(
        {
            "bamboogle": 10.0,
            "2wiki": 20.0,
            "hotpotqa": 30.0,
            "musique": 40.0,
            "gaia": 5.0,
        }
    )
    assert abs(m["avg"] - 25.0) < 1e-9
    assert m["gaia"] == 5.0
