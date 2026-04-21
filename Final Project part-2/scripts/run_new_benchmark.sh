#!/usr/bin/env bash
set -euo pipefail

python3 -m final_project_part2.new_benchmark.humaneval_runner \
  --dataset_path final_project_part2/new_benchmark/dataset_local/humaneval.jsonl \
  --output_json final_project_part2/outputs/new_benchmark/humaneval_qwen.json
