from dataclasses import dataclass
from typing import Any, Dict, Literal

ActionType = Literal["Search", "Identify", "Plan", "Edit", "Evaluate", "Finish"]


@dataclass
class Action:
    type: ActionType
    payload: Dict[str, Any]
