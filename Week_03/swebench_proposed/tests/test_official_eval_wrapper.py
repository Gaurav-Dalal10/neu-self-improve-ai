from pathlib import Path

import eval.official_swebench_runner as runner


class _Result:
    returncode = 0


def test_official_wrapper_builds_command(monkeypatch):
    captured = {}

    def _fake_run(cmd):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    rc = runner.run_official_eval(
        dataset_name="princeton-nlp/SWE-bench_Lite",
        predictions_path=Path("predictions.jsonl"),
        run_id="rid",
        max_workers=2,
    )

    assert rc == 0
    assert captured["cmd"][0:3] == ["python3", "-m", "swebench.harness.run_evaluation"]
    assert "--run_id" in captured["cmd"]
