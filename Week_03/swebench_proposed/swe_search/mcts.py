from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from swe_search.action_agent import ActionAgent
from swe_search.candidate_pool import CandidatePool, ScoredCandidate
from swe_search.node import Node
from swe_search.state import SearchState
from swe_search.value_agent import ValueAgent


@dataclass
class MCTSConfig:
    rollouts: int = 50
    max_depth: int = 10
    branching_k: int = 3
    C: float = 1.2
    alpha: float = 0.3
    beta: float = 0.7
    gamma: float = 0.03
    low_reward_prune_k: int = 3


class MCTSRunner:
    def __init__(
        self,
        config: MCTSConfig,
        action_agent: ActionAgent,
        value_agent: ValueAgent,
        tree_output_dir: Path,
    ):
        self.config = config
        self.action_agent = action_agent
        self.value_agent = value_agent
        self.tree_output_dir = tree_output_dir
        self.tree_output_dir.mkdir(parents=True, exist_ok=True)

    def _uct(self, parent: Node, child: Node) -> float:
        q = child.mean_value()
        u = self.config.C * math.sqrt(math.log(parent.visits + 1) / (child.visits + 1))
        d = max(1, child.depth)
        depth_term = self.config.alpha * math.exp(-self.config.beta * (d - 1)) - self.config.gamma * d
        return q + u + depth_term

    def _select(self, root: Node) -> Node:
        node = root
        while node.children and not node.is_terminal(self.config.max_depth):
            active = [c for c in node.children if not c.pruned]
            if not active:
                break
            node = max(active, key=lambda c: self._uct(node, c))
        return node

    def _expand(self, node: Node, instance: Dict[str, Any], run_id: str, rollout_idx: int) -> Node:
        action = self.action_agent.propose_action(node.state, node.hindsight_feedback)
        child_state = self.action_agent.simulate(instance, node.state, action, run_id=run_id, rollout_idx=rollout_idx)
        child = Node(
            state=child_state,
            parent=node,
            action_from_parent=action,
            depth=node.depth + 1,
        )
        node.children.append(child)
        return child

    def _backup(self, node: Node, value: float):
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value
            cur = cur.parent

    def _serialize_node(self, node: Node) -> Dict[str, Any]:
        return {
            "instance_id": node.state.instance_id,
            "depth": node.depth,
            "visits": node.visits,
            "value_sum": round(node.value_sum, 4),
            "mean_value": round(node.mean_value(), 4),
            "done": node.state.done,
            "patch_len": len(node.state.terminal_patch or ""),
            "children": [self._serialize_node(c) for c in node.children],
        }

    def run(self, root_state: SearchState, instance: Dict[str, Any], run_id: str) -> Tuple[Node, List[ScoredCandidate]]:
        root = Node(state=root_state)
        pool = CandidatePool(max_candidates=5)
        low_reward_streak = 0

        for rollout_idx in range(self.config.rollouts):
            leaf = self._select(root)
            if leaf.is_terminal(self.config.max_depth):
                value, explanation = self.value_agent.score(leaf.state)
                self._backup(leaf, value)
                pool.add(leaf.state, value, explanation)
                continue

            child = self._expand(leaf, instance=instance, run_id=run_id, rollout_idx=rollout_idx)
            value, explanation = self.value_agent.score(child.state)
            child.parent.hindsight_feedback.append(explanation)
            self._backup(child, value)
            pool.add(child.state, value, explanation)

            if value < 0.2:
                low_reward_streak += 1
            else:
                low_reward_streak = 0
            if low_reward_streak >= self.config.low_reward_prune_k:
                child.pruned = True

        tree_path = self.tree_output_dir / f"{root_state.instance_id.replace('/', '_')}.json"
        tree_path.write_text(json.dumps(self._serialize_node(root), indent=2), encoding="utf-8")
        return root, pool.top_diverse()
