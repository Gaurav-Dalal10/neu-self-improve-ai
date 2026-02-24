from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import AgentConfig, ProposedAgent
from database import DatabaseManager
from swe_search.actions import Action
from swe_search.state import SearchState


class ActionAgent:
    """Wrapper over existing ProposedAgent for MCTS expansion."""

    def __init__(
        self,
        db_path: str,
        model: str,
        max_steps_per_rollout: int,
        temperature: float = 0.2,
    ):
        self.db_path = db_path
        self.model = model
        self.max_steps_per_rollout = max_steps_per_rollout
        self.temperature = temperature

    def propose_action(self, state: SearchState, hindsight_feedback: List[str]) -> Action:
        return Action(
            type="Finish",
            payload={
                "feedback": hindsight_feedback[-3:],
                "max_steps": self.max_steps_per_rollout,
            },
        )

    def simulate(self, instance: Dict[str, Any], state: SearchState, action: Action, run_id: str, rollout_idx: int) -> SearchState:
        feedback = action.payload.get("feedback") or []

        instance_for_agent = dict(instance)
        if feedback:
            # Agent prompt builder consumes hints_text; append hindsight to preserve legacy behavior.
            hint = instance_for_agent.get("hints_text", "") or ""
            joined = "\n".join(f"- {line}" for line in feedback)
            instance_for_agent["hints_text"] = f"{hint}\n\nHindsight feedback:\n{joined}".strip()

        repo_path = Path("temp_repos") / f"{state.instance_id.replace('/', '_')}__{run_id}__r{rollout_idx}"

        with DatabaseManager(self.db_path) as db:
            config = AgentConfig(
                model=self.model,
                max_steps=self.max_steps_per_rollout,
                temperature=self.temperature,
            )
            agent = ProposedAgent(config, db)
            result = agent.solve_instance(instance_for_agent, str(repo_path))

        patch = result.get("patch") or ""
        terminal = bool(patch.strip())
        next_state = replace(
            state,
            repo_path=str(repo_path),
            done=terminal,
            terminal_patch=patch,
            context={
                **state.context,
                "trajectory_id": result.get("trajectory_id"),
                "rollout_success": bool(result.get("success")),
            },
            trajectory=[
                *state.trajectory,
                {
                    "action": action.type,
                    "feedback": feedback,
                    "patch_len": len(patch),
                    "trajectory_id": result.get("trajectory_id"),
                },
            ],
        )
        return next_state
