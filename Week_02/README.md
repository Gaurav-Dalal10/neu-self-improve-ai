# SWE-bench Baseline (Week 02)

**Group members: Mayank Bhadrasen, Gaurav Dalal, Srinjana Nag**

This project runs a SWE-bench Lite baseline agent and evaluates it with test execution.

## What this project contains

- `agent.py` - runs the tool-based agent and stores trajectories/patches.
- `load_data.py` - loads SWE-bench Lite instances into `swe_bench.db`.
- `eval/run.py` - evaluates generated patches by applying patch + running tests.
- `analyze.py` - prints aggregated metrics and can export CSV.
- `save_results.py` - saves a text summary.

## Metric definitions

- `resolved` / pass@1: `1` only when tests pass (exit code `0`) after patch apply.
- `patch_generated`: whether any patch/diff was produced.

Do not treat `patch_generated` as resolved.

## Setup

From `Week_02/swebench_baseline_Part1`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
"deactivate" to exit from your Virtual Environment
```

## Configure OpenAI API key

`.env` file

```bash
# create .env file and set OPENAI_API_KEY=your_key_here
# eg: 

# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4o
```

## Run the project 

1. Verify setup

```bash
python3 verify_setup.py
```

2. Load SWE-bench Lite data

```bash
python3 load_data.py
```

3. Run agent

```bash
python3 agent.py --limit=5

```

4. Run test-based evaluation

```bash
python3 -m eval.run --db swe_bench.db --split test --limit 5 --output_dir runs/quick_check
```

5. Analyze and save results

```bash
python3 analyze.py
python3 analyze.py --export
python3 save_results.py
```

## Where results are saved

If you used `--output_dir runs/quick_check`, results are here:

- `runs/quick_check/evaluation_results.csv`
- `runs/quick_check/summary.json`
- `runs/quick_check/<instance_id>/patch.diff`
- `runs/quick_check/<instance_id>/stdout.log`
- `runs/quick_check/<instance_id>/stderr.log`
- `runs/quick_check/<instance_id>/evaluation.json`

Additional exports:

- `results.csv` (from `analyze.py --export`)
- `results_summary.txt` (from `save_results.py`)

## Useful commands

Find latest run folder:

```bash
ls -1t runs | head -n 1
```

View summary quickly:

```bash
cat runs/quick_check/summary.json
cat results_summary.txt
```

Run tests for evaluation logic:

```bash
python3 -m pytest -q
```
