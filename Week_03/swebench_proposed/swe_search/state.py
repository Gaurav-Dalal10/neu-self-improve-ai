from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchState:
    instance_id: str
    repo: str
    base_commit: str
    repo_path: str
    done: bool = False
    terminal_patch: Optional[str] = None
    last_test_exit_code: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
