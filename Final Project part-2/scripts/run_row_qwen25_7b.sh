#!/usr/bin/env bash
set -euo pipefail

MODEL_LABEL="${MODEL_LABEL:-Qwen2.5-7B-Instruct}"
AGENTFLOW_DIR_ARG="${AGENTFLOW_DIR:-}"
AF_ARGS=()
if [[ -n "${AGENTFLOW_DIR_ARG}" ]]; then
  AF_ARGS+=(--agentflow_dir "${AGENTFLOW_DIR_ARG}")
fi

TASKS=(bamboogle 2wiki hotpotqa musique gaia)

set +e
python3 - <<'PY'
from final_project_part2.wrapper.env_setup import (
    assert_stdlib_not_shadowed,
    canonicalize_openai_env,
    resolve_agentflow_dir,
    validate_provider_env,
)
import os, sys
try:
    canonicalize_openai_env()
    assert_stdlib_not_shadowed()
    resolve_agentflow_dir(os.getenv("AGENTFLOW_DIR"))
    env_status = validate_provider_env()
    if not env_status["ok"]:
        print(f"[error] Missing required provider env vars: {env_status['missing']}")
        print("[hint] Required: OPENAI_API_KEY, OPENAI_BASE_URL")
        print("[hint] OPENAI_COMPAT_* aliases are accepted and auto-mapped.")
        sys.exit(2)
except Exception as e:
    print(f"[error] {e}")
    sys.exit(2)
PY
precheck_rc=$?
set -e
if [[ $precheck_rc -ne 0 ]]; then
  echo "[hint] Set AGENTFLOW_DIR to your local clone path, then rerun."
  exit $precheck_rc
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] 'uv' not found. Install with: brew install uv"
  exit 2
fi
if ! command -v parallel >/dev/null 2>&1; then
  echo "[error] 'parallel' not found. Install with: brew install parallel"
  exit 2
fi

fail_count=0
for t in "${TASKS[@]}"; do
  echo "[run] task=$t model=$MODEL_LABEL"
  cmd=(python3 -m final_project_part2.wrapper.run_upstream --task "$t" --model_label "$MODEL_LABEL" --output_dir final_project_part2/outputs/row_repro)
  if [[ ${#AF_ARGS[@]} -gt 0 ]]; then
    cmd=(python3 -m final_project_part2.wrapper.run_upstream --task "$t" --model_label "$MODEL_LABEL" "${AF_ARGS[@]}" --output_dir final_project_part2/outputs/row_repro)
  fi
  cmd+=(--strict)
  set +e
  "${cmd[@]}"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[warn] task $t failed (rc=$rc)."
    fail_count=$((fail_count+1))
  fi
 done

if [[ "$fail_count" -gt 0 ]]; then
  echo "[error] $fail_count task(s) failed. Check *_run.log files under final_project_part2/outputs/row_repro/$MODEL_LABEL/"
  exit 3
fi

sum_cmd=(python3 -m final_project_part2.wrapper.summarize --model_label "$MODEL_LABEL" --output_dir final_project_part2/outputs/row_repro)
if [[ ${#AF_ARGS[@]} -gt 0 ]]; then
  sum_cmd=(python3 -m final_project_part2.wrapper.summarize --model_label "$MODEL_LABEL" "${AF_ARGS[@]}" --output_dir final_project_part2/outputs/row_repro)
fi
"${sum_cmd[@]}"

python3 - <<'PY'
import json, pathlib
p=pathlib.Path('final_project_part2/outputs/row_repro')
subdirs=[d for d in p.iterdir() if d.is_dir()]
if not subdirs:
    raise SystemExit('No output dirs found')
latest=sorted(subdirs,key=lambda d:d.stat().st_mtime)[-1]
obj=json.loads((latest/'metrics.json').read_text())
if not obj.get('sources'):
    raise SystemExit("[error] No final_scores_direct_output.json files found. Benchmark execution did not produce results.")
print('\n=== Target Comparison ===')
for k in ['bamboogle','2wiki','hotpotqa','musique','avg','gaia']:
    ours=obj['metrics'].get(k,0.0)
    tgt=obj['targets'][k]
    print(f"{k:10s} ours={ours:.4f} target={tgt:.4f} delta={ours-tgt:+.4f}")
print(f"\nWrote: {latest/'metrics.json'}")
print(f"Wrote: {latest/'metrics.csv'}")
PY
