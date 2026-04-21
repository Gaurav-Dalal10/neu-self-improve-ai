from typing import Dict, List


def pass_at_1(passed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(passed) / float(total)


def build_summary(model: str, total: int, passed: int) -> Dict:
    return {
        "model": model,
        "total": total,
        "passed": passed,
        "pass@1": pass_at_1(passed, total),
    }
