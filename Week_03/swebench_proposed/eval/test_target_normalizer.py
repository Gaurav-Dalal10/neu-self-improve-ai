"""Utilities to normalize FAIL_TO_PASS targets into runnable pytest node ids."""

from __future__ import annotations

import re
from typing import Iterable, List

_PATTERN = re.compile(r"^(?P<name>[^\(]+)\s*\((?P<module>[\w\.]+)\)$")


def normalize_target(raw_target: str) -> str:
    t = (raw_target or "").strip()
    if not t:
        return ""

    # already pytest-style
    if "::" in t or "/" in t:
        return t

    m = _PATTERN.match(t)
    if not m:
        return t

    name = m.group("name").strip()
    mod = m.group("module").strip()
    parts = mod.split(".")
    if len(parts) >= 2:
        path = "/".join(parts[:-1]) + ".py"
        klass = parts[-1]
        return f"{path}::{klass}::{name}"
    return t


def normalize_targets(targets: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in targets:
        n = normalize_target(str(item))
        if n:
            out.append(n)
    return out
