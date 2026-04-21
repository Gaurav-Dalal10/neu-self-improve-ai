from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentMemory:
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, item: Dict[str, Any]) -> None:
        self.steps.append(item)

    def as_list(self) -> List[Dict[str, Any]]:
        return self.steps
