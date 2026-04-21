#!/usr/bin/env bash
set -euo pipefail

AGENTFLOW_DIR="${AGENTFLOW_DIR:-/Users/mayank/Desktop/neu-self-improve-ai/Final_Project/AgentFlow}"
MODEL="${1:-gpt-4o-mini}"
LABEL="${2:-Qwen2.5-7B-Instruct}"

python3 - "$AGENTFLOW_DIR" "$MODEL" "$LABEL" <<'PY'
from pathlib import Path
import sys

agentflow_dir = Path(sys.argv[1])
model = sys.argv[2]
label = sys.argv[3]
tasks = ["bamboogle", "2wiki", "hotpotqa", "musique", "gaia"]
new_models = f'''MODELS=(
    ":{model},{label},\\
Base_Generator_Tool|Python_Coder_Tool|Google_Search_Tool|Wikipedia_Search_Tool,\\
Default|Default|Default|Default,\\
trainable|trainable|trainable|trainable"
)
'''

for t in tasks:
    p = agentflow_dir / "test" / t / "run.sh"
    s = p.read_text(encoding="utf-8")
    a = s.find("MODELS=(")
    b = s.find(")\n\n# Get the directory where this script is located")
    if a == -1 or b == -1:
        raise RuntimeError(f"Could not patch MODELS block in {p}")
    p.write_text(s[:a] + new_models + s[b+2:], encoding="utf-8")
    print(f"patched {p}")
PY

echo "Patched all benchmark run.sh files to model='$MODEL' label='$LABEL'"
