# Results Summary

This file will be updated after real endpoint/dataset runs.

## Model Substitution Note (April 20, 2026)

Per professor guidance, substitutions are allowed when exact models are unavailable in Portkey.
Current plan uses available Qwen models from Model Catalog and maps required Qwen3.5 slots
to substitutes in `configs/qwen_models.yaml`.

## Step-1 Reproduction Model Note

- Intended assignment model: `Qwen-2.5-7B-Instruct` (AgentFlow no Flow-GRPO row).
- Current routed inference model (when Portkey Qwen routing is unavailable): `gpt-4o-mini`.
- Reason for substitution: provider availability/routing constraints in Portkey for required Qwen paths during this week, consistent with instructor allowance to substitute based on availability.
- Reporting rule: every table must include `model_used`, `target_model_slot`, and `substitution_reason`.

Planned sections:
- Target row vs reproduced row (Qwen-2.5-7B-Instruct, no Flow-GRPO)
- Qwen3.5 no-training sweep summary
- New benchmark summary
- Flow-GRPO LoRA trained vs untrained comparison
- Runtime/cost notes
