#!/usr/bin/env bash
set -euo pipefail

AGENTFLOW_DIR="${AGENTFLOW_DIR:-/Users/mayank/Desktop/neu-self-improve-ai/Final_Project/AgentFlow}"
MODEL="${MODEL:-gpt-4o-mini}"

cd "$AGENTFLOW_DIR/test"
uv run python solve.py \
  --index 0 \
  --task bamboogle \
  --data_file bamboogle/data/data.json \
  --llm_engine_name "$MODEL" \
  --root_cache_dir bamboogle/cache \
  --output_json_dir bamboogle/results/debug \
  --output_types direct \
  --enabled_tools "Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool" \
  --tool_engine "Default,Default,Default,Default" \
  --model_engine "trainable,trainable,trainable,trainable" \
  --max_time 120 \
  --max_steps 3 \
  --temperature 0.0

echo "Smoke run finished. Output: $AGENTFLOW_DIR/test/bamboogle/results/debug/output_0.json"
