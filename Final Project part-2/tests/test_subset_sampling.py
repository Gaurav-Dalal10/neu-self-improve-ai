from final_project_part2.agentflow_wrapper.eval.subsets import sample_examples


def test_subset_sampling_deterministic():
    rows = [{"i": i} for i in range(50)]
    a = sample_examples(rows, 10, 123)
    b = sample_examples(rows, 10, 123)
    assert [x["i"] for x in a] == [x["i"] for x in b]
