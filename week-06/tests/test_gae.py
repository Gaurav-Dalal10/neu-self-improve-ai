"""Verify GAE against a hand-computed example."""
import numpy as np
import torch
from rl.buffer import RolloutBuffer


def test_gae_reference():
    b = RolloutBuffer(T=3, N=1, obs_dim=2, act_dim=2,
                      device=torch.device("cpu"))
    b.rewards[:, 0] = [1.0, 2.0, 3.0]
    b.values[:, 0]  = [0.5, 1.0, 1.5]
    b.dones[:, 0]   = [0.0, 0.0, 1.0]

    b.compute_gae(
        next_value=np.array([0.0]),
        next_done=np.array([1.0]),
        gamma=0.99, lam=0.95,
    )

    # t=2: δ=3-1.5=1.5,                    A2=1.5
    # t=1: nnt=1-done[2]=0 → δ=2-1=1.0,    A1=1.0
    # t=0: δ=1+0.99*1-0.5=1.49,             A0=1.49+0.99*0.95*1.0=2.4305
    expected = np.array([2.4305, 1.0, 1.5], dtype=np.float32)
    np.testing.assert_allclose(b.advantages[:, 0], expected, atol=1e-4)
