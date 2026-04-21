# Reproducibility

## Determinism and artifacts

- Deterministic summary file discovery (exact path first, then latest fallback).
- Stable metric extraction rules from `final_scores_direct_output.json`.
- Unified outputs written to:
  - `final_project_part2/outputs/row_repro/<model_label>/metrics.json`
  - `final_project_part2/outputs/row_repro/<model_label>/metrics.csv`

## Phase 1 commands

1. Ensure AgentFlow repo exists locally:
```bash
git clone https://github.com/lupantech/AgentFlow.git
```

2. Verify benchmark data files:
```bash
ls -lh AgentFlow/test/bamboogle/data/data.json
ls -lh AgentFlow/test/2wiki/data/data.json
ls -lh AgentFlow/test/hotpotqa/data/data.json
ls -lh AgentFlow/test/musique/data/data.json
ls -lh AgentFlow/test/gaia/data/data.json
```

3. Optional schema check:
```bash
python3 - <<'PY'
import json, pathlib
paths = [
  'AgentFlow/test/bamboogle/data/data.json',
  'AgentFlow/test/2wiki/data/data.json',
  'AgentFlow/test/hotpotqa/data/data.json',
  'AgentFlow/test/musique/data/data.json',
  'AgentFlow/test/gaia/data/data.json',
]
for p in paths:
    pp = pathlib.Path(p)
    if not pp.exists():
        print(f'MISSING: {p}')
        continue
    obj = json.loads(pp.read_text(encoding='utf-8'))
    n = len(obj) if isinstance(obj, list) else len(obj.get('data', [])) if isinstance(obj, dict) else 0
    print(f'{p}: type={type(obj).__name__} size={n}')
PY
```

4. Run row reproduction wrapper:
```bash
export AGENTFLOW_DIR="/abs/path/to/AgentFlow"
source final_project_part2/scripts/setup_runtime_env.sh
export OPENAI_API_KEY="<PORTKEY_API_KEY>"
bash final_project_part2/scripts/patch_agentflow_openai_route.sh gpt-4o-mini Qwen2.5-7B-Instruct
bash final_project_part2/scripts/smoke_test_agentflow.sh
bash final_project_part2/scripts/run_row_qwen25_7b.sh
```

5. Verify benchmark score files:
```bash
find "$AGENTFLOW_DIR/test" -type f -name "final_scores_direct_output.json"
```

## Target comparison

Target row for no-training AgentFlow (assignment):
- Bamboogle=58.4, 2Wiki=60.0, HotpotQA=51.3, Musique=19.2, Avg=47.2, GAIA=17.2

The wrapper writes deltas vs targets into `metrics.json` and `metrics.csv`.
