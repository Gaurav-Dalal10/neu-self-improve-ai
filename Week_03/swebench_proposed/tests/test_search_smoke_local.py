from pathlib import Path

from swe_search.actions import Action
from swe_search.mcts import MCTSConfig, MCTSRunner
from swe_search.state import SearchState


class _FakeActionAgent:
    def propose_action(self, state, hindsight_feedback):
        return Action(type="Finish", payload={})

    def simulate(self, instance, state, action, run_id, rollout_idx):
        return SearchState(
            instance_id=state.instance_id,
            repo=state.repo,
            base_commit=state.base_commit,
            repo_path=state.repo_path,
            done=True,
            terminal_patch="diff --git a/a.py b/a.py\n",
        )


class _FakeValueAgent:
    def score(self, state):
        return (0.9, "likely good")


def test_search_smoke_produces_candidate(tmp_path: Path):
    runner = MCTSRunner(
        MCTSConfig(rollouts=3, max_depth=3),
        _FakeActionAgent(),
        _FakeValueAgent(),
        tmp_path,
    )
    root_state = SearchState("inst", "repo", "commit", "path")
    _root, candidates = runner.run(root_state, instance={"instance_id": "inst"}, run_id="smoke")
    assert candidates
