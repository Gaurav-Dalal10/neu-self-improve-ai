"""Verify env state save → perturb → restore recovers the original."""
import numpy as np
from rl.envs import make_env, save_state, restore_state


def test_roundtrip():
    env = make_env("MiniGrid-FourRooms-v0", seed=0)
    env.reset(seed=0)

    # advance a few steps
    for _ in range(5):
        env.step(env.action_space.sample())

    snap = save_state(env)

    # perturb further
    for _ in range(3):
        env.step(env.action_space.sample())

    restore_state(env, snap)
    u = env.unwrapped
    assert tuple(u.agent_pos) == tuple(snap["agent_pos"])
    assert int(u.agent_dir)   == snap["agent_dir"]
    assert np.array_equal(u.grid.encode(), snap["grid"])
    env.close()
