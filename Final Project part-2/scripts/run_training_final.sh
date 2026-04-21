#!/usr/bin/env bash
set -euo pipefail

python3 final_project_part2/training_modal/preflight.py --check_endpoint
python3 final_project_part2/training_modal/modal_train_flowgrpo_lora.py \
  --mode final \
  --require_endpoint \
  --backend mock
