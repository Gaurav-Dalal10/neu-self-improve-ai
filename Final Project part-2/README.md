# Final Project Part 2

Phase-1 implementation for wrapping upstream AgentFlow benchmark execution and deterministic score summarization.

## Current Phase

- Implemented: **Phase 1 wrapper + docs + tests**
- Pending: live provider runs (requires user inputs in `NEXT_STEPS_FROM_USER.md`)

## Required external repo

```bash
git clone https://github.com/lupantech/AgentFlow.git
```

AgentFlow benchmark datasets are expected in:

```bash
ls -lh AgentFlow/test/bamboogle/data/data.json
ls -lh AgentFlow/test/2wiki/data/data.json
ls -lh AgentFlow/test/hotpotqa/data/data.json
ls -lh AgentFlow/test/musique/data/data.json
ls -lh AgentFlow/test/gaia/data/data.json
```

## Run reproduction row (Qwen2.5-7B slot)

```bash
source final_project_part2/scripts/setup_runtime_env.sh
export OPENAI_API_KEY="<PORTKEY_API_KEY>"
export AGENTFLOW_DIR="/abs/path/to/AgentFlow"

# Optional: force all AgentFlow benchmark scripts to one model route
bash final_project_part2/scripts/patch_agentflow_openai_route.sh gpt-4o-mini Qwen2.5-7B-Instruct

# Optional: one-sample smoke test before full row run
bash final_project_part2/scripts/smoke_test_agentflow.sh

bash final_project_part2/scripts/run_row_qwen25_7b.sh
```

Outputs:
- `final_project_part2/outputs/row_repro/<model_label>/metrics.json`
- `final_project_part2/outputs/row_repro/<model_label>/metrics.csv`
- per-task logs in same directory

## Quick tests

```bash
python3 -m pytest -q final_project_part2/tests
```
