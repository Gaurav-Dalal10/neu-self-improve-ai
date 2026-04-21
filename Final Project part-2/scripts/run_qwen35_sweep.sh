#!/usr/bin/env bash
set -euo pipefail

python3 -m final_project_part2.agentflow_wrapper.eval.run_qwen35_sweep \
  --base_url "${OPENAI_BASE_URL:-${OPENAI_COMPAT_BASE_URL:-}}" \
  --api_key "${OPENAI_API_KEY:-${OPENAI_COMPAT_API_KEY:-}}" \
  --bamboogle_path "${BAMBOOGLE_PATH:-}" \
  --wiki2_path "${WIKI2_PATH:-}" \
  --hotpotqa_path "${HOTPOTQA_PATH:-}" \
  --musique_path "${MUSIQUE_PATH:-}" \
  --gaia_path "${GAIA_PATH:-}" \
  --output_csv final_project_part2/outputs/qwen35_no_training/summary.csv
