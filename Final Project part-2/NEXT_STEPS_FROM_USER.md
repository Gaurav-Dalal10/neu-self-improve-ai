# Next Steps From User

Phase 1 is implemented. Provide the following to execute live runs:

1. `AGENTFLOW_DIR`
- Absolute path to local AgentFlow clone.
- Must contain `test/<task>/run.sh` and `agentflow/.env.template`.

2. Provider endpoint choice and env vars
- Portkey option:
  - `OPENAI_COMPAT_BASE_URL=https://api.portkey.ai/v1`
  - `OPENAI_COMPAT_API_KEY=<PORTKEY_API_KEY>`
- or DashScope:
  - `DASHSCOPE_API_KEY=<...>`
- or Together:
  - `TOGETHER_API_KEY=<...>`

3. Exact model routing strings
- Qwen2.5-7B-Instruct route string used by your provider.
- Qwen3.5 size slots (0.8B/2B/4B/9B/27B) route strings.
- If unavailable, confirm substitutions to use from `configs/models.yaml`.

4. New benchmark dataset
- Local file path and schema for chosen benchmark (codegen or text-to-sql).

5. Modal settings for later training phases
- Secret names for provider/API keys.
- Volume names for checkpoints/cache.
- Confirm GPU choice (`L40S` preferred, `A100-40` fallback).

6. Experiment policy
- Number of seeds (1/2/3).
- Any benchmark sample-size overrides.
