from swe_search.candidate_pool import CandidatePool
from swe_search.state import SearchState


def _state(instance_id: str, patch: str):
    return SearchState(
        instance_id=instance_id,
        repo="r",
        base_commit="c",
        repo_path="p",
        done=True,
        terminal_patch=patch,
    )


def test_candidate_pool_limits_and_filters():
    pool = CandidatePool(max_candidates=2)
    pool.add(_state("i1", "diff --git a/a.py b/a.py\n"), 0.9, "a")
    pool.add(_state("i2", "diff --git a/b.py b/b.py\n"), 0.8, "b")
    pool.add(_state("i3", "diff --git a/c.py b/c.py\n"), 0.7, "c")

    out = pool.top_diverse()
    assert len(out) == 2
    assert out[0].score >= out[1].score
