"""Verify the room-based diversity metric."""
from rl.envs import diversity


def test_all_same():
    sets = [frozenset([0])] * 4
    assert diversity(sets, n=4) == 0.25   # 1 unique / 4


def test_all_different():
    sets = [frozenset([0]), frozenset([1]), frozenset([2]), frozenset([3])]
    assert abs(diversity(sets, n=4) - 1.0) < 1e-8


def test_partial():
    sets = [frozenset([0]), frozenset([0]), frozenset([1]), frozenset([2])]
    assert abs(diversity(sets, n=4) - 0.75) < 1e-8


def test_empty():
    assert diversity([], n=0) == 0.0
