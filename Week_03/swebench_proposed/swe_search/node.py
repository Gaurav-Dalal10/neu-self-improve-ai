from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from swe_search.actions import Action
from swe_search.state import SearchState


@dataclass
class Node:
    state: SearchState
    parent: Optional["Node"] = None
    action_from_parent: Optional[Action] = None
    depth: int = 0
    visits: int = 0
    value_sum: float = 0.0
    children: List["Node"] = field(default_factory=list)
    hindsight_feedback: List[str] = field(default_factory=list)
    pruned: bool = False

    def mean_value(self) -> float:
        return self.value_sum / max(self.visits, 1)

    def is_terminal(self, max_depth: int) -> bool:
        return self.state.done or self.depth >= max_depth
