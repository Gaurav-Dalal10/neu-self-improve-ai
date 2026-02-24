# SWE-bench Proposed (Week 03)

**Group members: Mayank Bhadrasen, Gaurav Dalal, Srinjana Nag**

Week 03 implements a proposed-search pipeline for SWE-bench Lite with:
- MCTS-style search over candidate fixes
- Value scoring with hindsight feedback
- Discriminator selection across top candidates
- Official SWE-bench harness integration for benchmark-grade evaluation

## Quickstart (GitHub/Grader)

```bash
cd Week_03/swebench_proposed
source .venv311/bin/activate
python3 -m swe_search.run --config configs/swe_search.yaml --run_id smoke10 --limit 10
python3 scripts/build_predictions.py --run_dir runs/smoke10 --out runs/smoke10/predictions.jsonl
python3 -m eval.official_swebench_runner --dataset_name princeton-nlp/SWE-bench_Lite --predictions_path runs/smoke10/predictions.jsonl --max_workers 1 --run_id smoke10_eval
```

Then:

```bash
mv gpt-4o.smoke10_eval.json runs/smoke10_eval/
python3 scripts/summarize_official_eval.py --eval_dir runs/smoke10_eval --out runs/smoke10_eval/summary.json
cat runs/smoke10_eval/summary.json
```

## Wins in this iteration

- **Official harness run completed successfully** on a smoke subset.
  - Submitted: **10**
  - Completed: **10**
  - Resolved: **2**
  - Unresolved: **8**
  - Errors: **0**
  - Empty patches: **0**
  - Artifact: `runs/smoke10_eval/gpt-4o.smoke10_eval.json`
- **Test suite expanded and passing**: **12/12 tests passed**.
- **New proposed pipeline package delivered**: **11 modules** under `swe_search/`.
- **Evaluation workflow upgraded**:
  - Official harness wrapper added
  - Prediction JSONL builder added
  - Summary parser updated for official schema outputs
  - FAIL_TO_PASS normalization utility added

## Project structure

### Core runtime
- `swe_search/run.py` - main proposed pipeline entrypoint
- `swe_search/mcts.py` - MCTS loop and UCT scoring
- `swe_search/action_agent.py` - wrapper over existing agent for search expansion
- `swe_search/value_agent.py` - value scoring + explanation feedback
- `swe_search/discriminator.py` - final candidate selection
- `swe_search/candidate_pool.py` - top-k diverse candidates
- `swe_search/state.py`, `swe_search/actions.py`, `swe_search/node.py`, `swe_search/reward.py`

### Evaluation
- `eval/official_swebench_runner.py` - wrapper around official `swebench.harness.run_evaluation`
- `scripts/build_predictions.py` - creates official predictions JSONL
- `scripts/summarize_official_eval.py` - summarizes official eval artifacts
- `eval/run.py` - local lite evaluator (diagnostic)
- `eval/harness.py` - local test-based harness

### Data and analysis
- `load_data.py` - loads SWE-bench Lite into `swe_bench.db`
- `analyze.py` - local run analysis
- `save_results.py` - local summary export

## Environment setup

From `Week_03/swebench_proposed`:

```bash
# Recommended for official harness compatibility
python3.11 -m venv .venv311
source .venv311/bin/activate

python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install swebench datasets
```

Configure `.env`:

```bash
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o
```

Docker is required for official SWE-bench harness:

```bash
docker version
docker info
```

## How to run

## 1) Load dataset

```bash
python3 load_data.py
```

## 2) Run proposed search pipeline

### Smoke (10 instances)

```bash
python3 -m swe_search.run --config configs/swe_search.yaml --run_id smoke10 --limit 10
```

### Main milestone (50 instances)

```bash
python3 -m swe_search.run --config configs/swe_search.yaml --run_id part2_lite50
```

## 3) Build predictions JSONL

```bash
python3 scripts/build_predictions.py --run_dir runs/smoke10 --out runs/smoke10/predictions.jsonl
```

(For 50-run, replace `smoke10` with `part2_lite50`.)

## 4) Run official SWE-bench evaluation

```bash
python3 -m eval.official_swebench_runner \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path runs/smoke10/predictions.jsonl \
  --max_workers 1 \
  --run_id smoke10_eval
```

If official harness writes report in project root, move it:

```bash
mv gpt-4o.smoke10_eval.json runs/smoke10_eval/
```

## 5) Summarize official results

```bash
python3 scripts/summarize_official_eval.py \
  --eval_dir runs/smoke10_eval \
  --out runs/smoke10_eval/summary.json

cat runs/smoke10_eval/summary.json
```

## 6) Run local tests

```bash
python3 -m pytest -q tests
```

## Useful outputs

- Search outputs: `runs/<run_id>/search_outputs.json`
- Search tree logs: `runs/<run_id>/tree/*.json`
- Official predictions: `runs/<run_id>/predictions.jsonl`
- Official eval report: `runs/<run_id>_eval/gpt-4o.<run_id>_eval.json`
- Official eval summary: `runs/<run_id>_eval/summary.json`

## Config

Primary config file:
- `configs/swe_search.yaml`

Default key settings:
- dataset limit: 50
- rollouts: 50
- max depth: 10
- branching_k: 3
- low_reward_prune_k: 3

## Notes

- `agent.py` is kept for compatibility with the legacy/local flow.
- `eval/run.py` remains useful for fast local diagnostics.
- Official benchmark reporting should use the official harness path described above.
