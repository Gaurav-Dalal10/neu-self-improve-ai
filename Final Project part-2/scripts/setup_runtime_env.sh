#!/usr/bin/env bash

_is_sourced() {
  # Bash
  if [ -n "${BASH_VERSION:-}" ]; then
    [ "${BASH_SOURCE[0]:-}" != "$0" ] && return 0 || return 1
  fi
  # Zsh
  if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in
      *:file) return 0 ;;
    esac
    return 1
  fi
  # Fallback
  return 1
}

if ! _is_sourced; then
  echo "Run with: source final_project_part2/scripts/setup_runtime_env.sh"
  exit 2
fi

export AGENTFLOW_DIR="${AGENTFLOW_DIR:-/Users/mayank/Desktop/neu-self-improve-ai/Final_Project/AgentFlow}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_COMPAT_BASE_URL:-https://api.portkey.ai/v1}}"

if [[ -n "${OPENAI_COMPAT_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${OPENAI_COMPAT_API_KEY}"
fi
if [[ -n "${OPENAI_API_KEY:-}" && -z "${OPENAI_COMPAT_API_KEY:-}" ]]; then
  export OPENAI_COMPAT_API_KEY="${OPENAI_API_KEY}"
fi
if [[ -n "${OPENAI_BASE_URL:-}" && -z "${OPENAI_COMPAT_BASE_URL:-}" ]]; then
  export OPENAI_COMPAT_BASE_URL="${OPENAI_BASE_URL}"
fi

unset PYTHONHOME 2>/dev/null || true
unset PYTHONPATH 2>/dev/null || true
export PYTHONPATH="${AGENTFLOW_DIR}"

echo "AGENTFLOW_DIR=$AGENTFLOW_DIR"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "OPENAI_API_KEY=${OPENAI_API_KEY:+set}"
python3 - <<'PY'
import types
print("types_module:", types.__file__)
PY
