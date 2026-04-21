# Training Playbook (Professor Constraints Aligned)

## Policy mapping

1. Stick close to paper; allow justified adjustments.
- We support subsetting and deterministic seeds.
- All outputs include mode/seed/config metadata.

2. Training budget controls.
- Dev runs are hard-capped at 600s.
- Final runs are hard-capped at 21600s (6h).
- Per-minute heartbeat emitted during training launcher.
- Early stop trigger available for stagnation.

3. Inference via provider endpoint.
- Preflight checks require endpoint env vars:
  - `OPENAI_COMPAT_BASE_URL`
  - `OPENAI_COMPAT_API_KEY`

## Recommended usage order

1. `bash final_project_part2/scripts/check_training_readiness.sh`
2. `bash final_project_part2/scripts/run_training_dev.sh`
3. Inspect `final_project_part2/training_modal/checkpoints/run_summary_dev.json`
4. Only after dev success, launch final mode.

## Output files

- `metrics_dev.jsonl` / `metrics_final.jsonl`
- `run_summary_dev.json` / `run_summary_final.json`
- `preflight_report.json`

