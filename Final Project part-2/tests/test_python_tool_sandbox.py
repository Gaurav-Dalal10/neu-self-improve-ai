from final_project_part2.agentflow_wrapper.tools.python_tool import run_python


def test_python_tool_sandbox_ok_and_timeout():
    ok = run_python("print(2+3)")
    assert ok["status"] == "ok"
    assert "5" in ok["stdout"]

    timeout = run_python("import time\ntime.sleep(2)\nprint('x')", timeout_sec=1)
    assert timeout["status"] == "timeout"
