import json
import subprocess
from pathlib import Path


def test_build_predictions_schema(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    source = run_dir / "search_outputs.json"
    source.write_text(
        json.dumps([
            {
                "instance_id": "i1",
                "model_name_or_path": "m",
                "model_patch": "diff --git a/a.py b/a.py\\n",
            }
        ]),
        encoding="utf-8",
    )
    out = tmp_path / "predictions.jsonl"
    subprocess.run(
        [
            "python3",
            "scripts/build_predictions.py",
            "--run_dir",
            str(run_dir),
            "--out",
            str(out),
        ],
        check=True,
    )
    line = out.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert set(payload.keys()) == {"instance_id", "model_name_or_path", "model_patch"}
