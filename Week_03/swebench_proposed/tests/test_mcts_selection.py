import math
from pathlib import Path

from swe_search.actions import Action
from swe_search.mcts import MCTSConfig, MCTSRunner
from swe_search.node import Node
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
            trajectory=state.trajectory + [{"rollout": rollout_idx}],
        )


class _FakeValueAgent:
    def score(self, state):
        return (0.5, "ok")


def test_uct_prefers_higher_value_when_visits_equal(tmp_path: Path):
    cfg = MCTSConfig()
    runner = MCTSRunner(cfg, _FakeActionAgent(), _FakeValueAgent(), tmp_path)

    root = Node(
        state=SearchState("i", "r", "c", "p"),
        visits=20,
    )
    child_a = Node(state=SearchState("i", "r", "c", "p"), parent=root, depth=1, visits=5, value_sum=4.5)
    child_b = Node(state=SearchState("i", "r", "c", "p"), parent=root, depth=1, visits=5, value_sum=1.0)
    score_a = runner._uct(root, child_a)
    score_b = runner._uct(root, child_b)
    assert score_a > score_b
