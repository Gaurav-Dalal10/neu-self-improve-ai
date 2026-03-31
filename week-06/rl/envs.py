"""
Environment helpers for MiniGrid Four Rooms.

- Observation wrapper: flattens the (7,7,3) grid image + one-hot direction → flat vector
- Vectorised constructor: wraps N envs in SyncVectorEnv
- State save / restore: checkpoint and restore MiniGrid internal state
  (used by vine sampling in Poly-PPO so we can branch from a saved state)
- Room-based diversity metric used by the polychromic objective
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid


# ─────────────────────────────────────────────
# Observation wrapper
# ─────────────────────────────────────────────

class FlatObsWrapper(gym.ObservationWrapper):
    """Flatten MiniGrid dict obs → single float32 vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        img = env.observation_space["image"]
        self._flat_dim = int(np.prod(img.shape)) + 4  # image + direction one-hot
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._flat_dim,), dtype=np.float32,
        )

    def observation(self, obs: Dict[str, Any]) -> np.ndarray:
        img = obs["image"].astype(np.float32).ravel()
        img /= max(img.max(), 1.0)          # normalise to [0,1]
        d = np.zeros(4, dtype=np.float32)
        d[int(obs.get("direction", 0)) % 4] = 1.0
        return np.concatenate([img, d])


# ─────────────────────────────────────────────
# Env constructors
# ─────────────────────────────────────────────

def make_env(env_id: str, seed: int, idx: int = 0,
             max_steps: int | None = None) -> gym.Env:
    """Create a single wrapped MiniGrid env."""
    env = gym.make(env_id)
    if max_steps is not None and hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = max_steps
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = FlatObsWrapper(env)
    env.reset(seed=seed + idx)
    return env


def make_vec_env(env_id: str, seed: int, n: int,
                 max_steps: int | None = None) -> gym.vector.SyncVectorEnv:
    """Create a synchronous vectorised env with *n* copies."""
    fns = [
        (lambda i=i: make_env(env_id, seed=seed, idx=i, max_steps=max_steps))
        for i in range(n)
    ]
    return gym.vector.SyncVectorEnv(fns)


# ─────────────────────────────────────────────
# State save / restore  (for vine sampling)
# ─────────────────────────────────────────────

def save_state(env: gym.Env) -> Dict[str, Any]:
    """Snapshot the full internal MiniGrid state."""
    u = env.unwrapped
    return {
        "grid":      np.array(u.grid.encode(), copy=True),
        "agent_pos": tuple(int(x) for x in u.agent_pos),
        "agent_dir": int(u.agent_dir),
        "step_count": int(getattr(u, "step_count", 0)),
    }


def restore_state(env: gym.Env, snap: Dict[str, Any]) -> None:
    """Overwrite the MiniGrid internal state from a snapshot."""
    u = env.unwrapped
    grid, _ = Grid.decode(np.array(snap["grid"], copy=True))
    u.grid       = grid
    u.agent_pos  = np.array(snap["agent_pos"], dtype=np.int64)
    u.agent_dir  = int(snap["agent_dir"])
    u.step_count = int(snap.get("step_count", 0))
    u.carrying   = None                       # Four Rooms never carries


def obs_from_env(env: gym.Env) -> np.ndarray:
    """Re-generate the flat observation from the current env state."""
    raw = env.unwrapped.gen_obs()
    return env.observation(raw)               # goes through FlatObsWrapper


# ─────────────────────────────────────────────
# Room identification & diversity metric
# ─────────────────────────────────────────────

def room_id(pos: Tuple[int, int], w: int, h: int) -> int:
    """Map an (x, y) grid position to a room index 0-3."""
    x, y = int(pos[0]), int(pos[1])
    return (0 if x < w // 2 else 1) + (0 if y < h // 2 else 2)


def rooms_visited(positions: List[Tuple[int, int]],
                  w: int, h: int) -> frozenset:
    """Return the set of rooms touched by a trajectory."""
    return frozenset(room_id(p, w, h) for p in positions)


def diversity(room_sets: List[frozenset], n: int) -> float:
    """
    Fraction of semantically distinct trajectories in a set.
    Two trajectories are distinct iff they visit different room-sets.
    Returns 0 when all trajectories visit the same rooms.
    """
    if n <= 0:
        return 0.0
    return float(np.clip(len(set(room_sets)) / n, 0.0, 1.0))
