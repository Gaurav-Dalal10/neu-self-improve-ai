from pathlib import Path

from swe_search.actions import Action
from swe_search.mcts import MCTSConfig, MCTSRunner
from swe_search.state import SearchState


class _FakeActionAgent:
    def propose_action(self, state, hindsight_feedback):
        return Action(type="Finish", payload={"feedback": hindsight_feedback})

    def simulate(self, instance, state, action, run_id, rollout_idx):
        return SearchState(
            instance_id=state.instance_id,
            repo=state.repo,
            base_commit=state.base_commit,
            repo_path=state.repo_path,
            done=True,
            terminal_patch=f"diff --git a/f{rollout_idx}.py b/f{rollout_idx}.py\n",
            trajectory=state.trajectory + [{"feedback_seen": list(action.payload.get("feedback") or [])}],
        )


class _FakeValueAgent:
    def score(self, state):
        return (0.6, "hindsight: tighten fix")


def test_hindsight_feedback_is_recorded(tmp_path: Path):
    cfg = MCTSConfig(rollouts=2, max_depth=3)
    runner = MCTSRunner(cfg, _FakeActionAgent(), _FakeValueAgent(), tmp_path)
    root_state = SearchState("inst", "repo", "commit", "path")

    root, _ = runner.run(root_state, instance={"instance_id": "inst"}, run_id="r1")
    assert root.hindsight_feedback
    assert "hindsight" in root.hindsight_feedback[0]
