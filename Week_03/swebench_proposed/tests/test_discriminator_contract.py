from swe_search.candidate_pool import ScoredCandidate
from swe_search.discriminator import Discriminator
from swe_search.state import SearchState


def _cand(score: float):
    state = SearchState(
        instance_id="i",
        repo="r",
        base_commit="c",
        repo_path="p",
        done=True,
        terminal_patch="diff --git a/a.py b/a.py\n",
    )
    return ScoredCandidate(state=state, score=score, explanation="x")


def test_discriminator_fallback_is_deterministic_without_llm():
    disc = Discriminator(use_llm=False)
    best = disc.choose([_cand(0.2), _cand(0.8), _cand(0.5)])
    assert best.score == 0.8
