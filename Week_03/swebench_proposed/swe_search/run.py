from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from database import DatabaseManager
from swe_search.action_agent import ActionAgent
from swe_search.discriminator import Discriminator
from swe_search.mcts import MCTSConfig, MCTSRunner
from swe_search.state import SearchState
from swe_search.value_agent import ValueAgent

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("PyYAML is required for swe_search.run. Install pyyaml>=6.0") from exc


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_instances(db: DatabaseManager, limit: int) -> List[Dict[str, Any]]:
    return db.get_all_instances(limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWE-Search style MCTS pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run_id", default=None, help="Run identifier")
    parser.add_argument("--limit", type=int, default=None, help="Override dataset limit")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    tree_dir = run_dir / "tree"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_limit = args.limit or int(config["dataset"]["limit"])

    mcts_cfg = MCTSConfig(**config["mcts"])
    action_agent = ActionAgent(
        db_path="swe_bench.db",
        model=config["models"]["action_model"],
        max_steps_per_rollout=int(config["execution"]["max_steps_per_action_rollout"]),
    )
    value_agent = ValueAgent(model=config["models"]["value_model"])
    discriminator = Discriminator(model=config["models"]["discriminator_model"])
    runner = MCTSRunner(
        config=mcts_cfg,
        action_agent=action_agent,
        value_agent=value_agent,
        tree_output_dir=tree_dir,
    )

    outputs: List[Dict[str, Any]] = []

    with DatabaseManager("swe_bench.db") as db:
        instances = _load_instances(db, limit=dataset_limit)
        if not instances:
            raise RuntimeError("No instances found in DB. Run load_data.py first.")

        for idx, instance in enumerate(instances, 1):
            instance_id = instance["instance_id"]
            print(f"[{idx}/{len(instances)}] MCTS search for {instance_id}")
            root_state = SearchState(
                instance_id=instance_id,
                repo=instance["repo"],
                base_commit=instance["base_commit"],
                repo_path="",
            )
            _root, candidates = runner.run(root_state, instance=instance, run_id=run_id)
            if not candidates:
                outputs.append(
                    {
                        "instance_id": instance_id,
                        "model_name_or_path": config["models"]["action_model"],
                        "model_patch": "",
                        "selection_reason": "no_candidates",
                    }
                )
                continue

            winner = discriminator.choose(candidates)
            outputs.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": config["models"]["action_model"],
                    "model_patch": winner.state.terminal_patch or "",
                    "selection_reason": "discriminator" if len(candidates) > 1 else "single_candidate",
                    "candidate_count": len(candidates),
                    "value_score": winner.score,
                    "value_explanation": winner.explanation,
                }
            )

    output_json = run_dir / "search_outputs.json"
    output_json.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
