from swe_search.state import SearchState


def compute_reward(state: SearchState) -> float:
    """Heuristic reward for search guidance before official evaluation."""
    patch = (state.terminal_patch or "").strip()
    if not patch:
        return 0.0
    if state.last_test_exit_code == 0:
        return 1.0
    if state.last_test_exit_code is None:
        return 0.4
    if state.last_test_exit_code == 124:
        return 0.2
    return 0.3
