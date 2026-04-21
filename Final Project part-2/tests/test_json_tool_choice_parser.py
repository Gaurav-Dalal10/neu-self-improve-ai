from final_project_part2.agentflow_wrapper.agent.planner import parse_tool_choice


def test_json_tool_choice_parser():
    out = parse_tool_choice('{"tool":"python_coder","subgoal":"calc","command":"print(2+2)"}', ["python_coder", "base_generator"])
    assert out["tool"] == "python_coder"
    assert out["subgoal"] == "calc"

    bad = parse_tool_choice('not-json', ["base_generator"])
    assert bad["tool"] == "base_generator"
