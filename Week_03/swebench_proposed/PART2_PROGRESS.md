# Part 2 Progress Log (Week_03)


### Scope 
- Goal: reproduce the project’s proposed-approach performance for Part 2.

### Core runtime modules:
  - `agent.py` (LLM-driven tool agent)
  - `tools.py` (repo search/read/edit/diff/test tools)
  - `database.py` (SQLite schema + persistence)
  - `load_data.py` (SWE-bench Lite dataset ingestion)
  - `eval/harness.py` + `eval/run.py` (test-based evaluation)
  - `analyze.py` + `save_results.py` (aggregation/export)
- Tests:
  - `tests/test_eval_harness.py` (evaluation semantics)
- Docs:
  - `README.md`, `ARCHITECTURE.md`
- Existing artifacts:
  - `swe_bench.db` with `300` instances and `5` trajectories
  - `runs/quick_check` with `5` evaluated runs

### Proposed-approach indicators found
- `ARCHITECTURE.md` documents a **Moatless-Adapted proposed approach** and explicitly excludes MCTS/value/discriminator.
- `ARCHITECTURE.md` claims: "successfully replicates ... with 77.78% resolve rate".

### Current assumption set
- Assumption A1: For this repo snapshot, "proposed approach" refers to the implemented Moatless-Adapted pipeline in `agent.py` + `eval/*`.
- Assumption A2: The `77.78%` value in `ARCHITECTURE.md` is the only explicit target present in-repo unless a stronger source is found later.
- Assumption A3: Reproduction should prioritize **test-based resolved/pass@1** from `eval/run.py` over legacy patch-generated metric.

### First reproduction command sequence (draft)
1. `source .venv/bin/activate`
2. `python3 verify_setup.py`
3. `python3 load_data.py` (idempotent refresh)
4. `python3 agent.py --limit=5`
5. `python3 -m eval.run --db swe_bench.db --split test --limit 5 --output_dir runs/part2_repro_limit5`
6. `python3 analyze.py`
7. `python3 save_results.py`

### Notes
- Existing run indicates patch-apply failures in evaluation (`error_type=patch_apply_failed`) for the first 5 astropy instances.
- If this reproduces again, inspect patch generation and evaluator apply semantics before changing algorithmic behavior.

### Verification outcome
- `python3 verify_setup.py`:
  - Dependencies: PASS
  - Environment: PASS
  - Database: PASS
  - OpenAI API connection: PASS

## 2026-02-22 - Proposed-result verification after user rerun

### Evidence reviewed
- `runs/part2_repro_limit5/summary.json` (generated_at `2026-02-22T02:28:26.987996Z`)
- `results_summary.txt`
- `evaluation_runs` rows in `swe_bench.db`
- Claimed target in `ARCHITECTURE.md`

### Additional note
- DB query for `eval_run_name='part2_repro_limit5'` currently returns 10 rows (run name reused twice); latest `summary.json` reflects the newest 5-row execution.

## 2026-02-22 - Reproduction blocker fixes applied

### Fix 1: patch apply failures due relative patch paths
- Issue: evaluator wrote patch files under relative output dirs (e.g., `runs/...`) and invoked `git apply` from cloned repo cwd, so patch path was unresolved and all runs became `patch_apply_failed`.
- Change: in `eval/harness.py`
  - resolve `output_dir` to absolute path at evaluation start
  - pass `patch_path.resolve()` to `git apply`

### Fix 2: evaluation selecting stale trajectories
- Issue: evaluator loaded trajectories with `ORDER BY t.timestamp ASC`, so repeated runs kept re-evaluating oldest trajectories.
- Change: in `eval/run.py`
  - switched ordering to `DESC`
  - default now evaluates latest trajectory per instance
  - added `--all_trajectories` to evaluate every matching trajectory

### Fix 3: latest trajectory may be empty patch
- Issue: latest trajectory can have empty `final_patch` even if a recent non-empty patch exists.
- Change: in `eval/run.py`
  - default now prefers latest non-empty patch per instance
  - added `--allow_empty_latest` to disable this preference

### Regression test
- Added `tests/test_eval_harness.py::test_relative_output_dir_still_applies_patch`
  to ensure relative output dirs do not cause patch-apply failures.

### Validation
- `python3 -m pytest -q tests` -> 4 passed.

## 2026-02-22 - Proposed pipeline implementation pass

### Objective
- Implement minimal necessary project changes to execute a paper-aligned proposed approach path (MCTS + value feedback + discriminator) and official SWE-bench harness integration.

### Added new package
- `swe_search/`
  - `actions.py`, `state.py`, `node.py`, `reward.py`
  - `action_agent.py` (wrapper over existing `ProposedAgent`)
  - `value_agent.py`
  - `candidate_pool.py`
  - `discriminator.py`
  - `mcts.py`
  - `run.py`
  - `__init__.py`

### Added evaluation + scripts
- `eval/official_swebench_runner.py` (official harness wrapper)
- `eval/test_target_normalizer.py` (FAIL_TO_PASS normalization utility)
- `scripts/build_predictions.py`
- `scripts/summarize_official_eval.py`
- `configs/swe_search.yaml`

### Updated existing files
- `eval/harness.py` now normalizes FAIL_TO_PASS targets before building pytest command.
- `requirements.txt` added `pyyaml>=6.0`.
- `README.md` includes proposed pipeline and official eval commands.

### Added tests
- `tests/test_mcts_selection.py`
- `tests/test_value_feedback.py`
- `tests/test_candidate_pool.py`
- `tests/test_discriminator_contract.py`
- `tests/test_search_smoke_local.py`
- `tests/test_predictions_schema.py`
- `tests/test_official_eval_wrapper.py`
- `tests/test_test_target_normalizer.py`
