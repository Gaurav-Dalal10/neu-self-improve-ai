import json
from pathlib import Path

from final_project_part2.wrapper.summarize import find_final_score_file, parse_accuracy_from_final_scores


def test_result_file_discovery_and_parsing(tmp_path: Path):
    af = tmp_path / "AgentFlow"
    score_path = af / "test" / "bamboogle" / "results" / "ModelX" / "final_scores_direct_output.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps({"accuracy": 61.5}), encoding="utf-8")

    found = find_final_score_file(af, "bamboogle", "ModelX")
    assert found == score_path
    assert abs(parse_accuracy_from_final_scores(found) - 61.5) < 1e-9
