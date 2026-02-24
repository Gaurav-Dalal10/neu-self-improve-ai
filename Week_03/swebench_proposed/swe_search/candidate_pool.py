from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from swe_search.state import SearchState


@dataclass
class ScoredCandidate:
    state: SearchState
    score: float
    explanation: str


def _changed_files(patch: str) -> set[str]:
    files = set()
    for line in patch.splitlines():
        if line.startswith("diff --git a/"):
            part = line.split("diff --git a/", 1)[1]
            files.add(part.split(" b/", 1)[0].strip())
    return files


class CandidatePool:
    def __init__(self, max_candidates: int = 5):
        self.max_candidates = max_candidates
        self._items: List[ScoredCandidate] = []

    def add(self, state: SearchState, score: float, explanation: str):
        if not (state.done and (state.terminal_patch or "").strip()):
            return
        self._items.append(ScoredCandidate(state=state, score=score, explanation=explanation))

    def top_diverse(self) -> List[ScoredCandidate]:
        ranked = sorted(self._items, key=lambda c: c.score, reverse=True)
        chosen: List[ScoredCandidate] = []
        chosen_files: List[set[str]] = []

        for cand in ranked:
            files = _changed_files(cand.state.terminal_patch or "")
            is_diverse = True
            for existing in chosen_files:
                if files and existing and len(files & existing) / len(files | existing) > 0.8:
                    is_diverse = False
                    break
            if is_diverse or len(chosen) < 2:
                chosen.append(cand)
                chosen_files.append(files)
            if len(chosen) >= self.max_candidates:
                break
        return chosen
