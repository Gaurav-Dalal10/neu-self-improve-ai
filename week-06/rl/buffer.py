"""
Rollout buffer that stores one rollout phase of shape [T, N]
(T timesteps × N parallel envs) and computes GAE advantages.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MiniBatch:
    """One minibatch yielded to the PPO update loop."""
    obs:        torch.Tensor
    actions:    torch.Tensor
    logprobs:   torch.Tensor
    advantages: torch.Tensor
    returns:    torch.Tensor
    values:     torch.Tensor
    old_logits: torch.Tensor


class RolloutBuffer:
    """Fixed-size numpy buffer filled step-by-step, then consumed in minibatches."""

    def __init__(self, T: int, N: int, obs_dim: int, act_dim: int,
                 device: torch.device):
        self.T, self.N = T, N
        self.device = device
        # Pre-allocate all arrays  [T, N, ...]
        self.obs        = np.zeros((T, N, obs_dim),  dtype=np.float32)
        self.actions    = np.zeros((T, N),            dtype=np.int64)
        self.logprobs   = np.zeros((T, N),            dtype=np.float32)
        self.rewards    = np.zeros((T, N),            dtype=np.float32)
        self.dones      = np.zeros((T, N),            dtype=np.float32)
        self.values     = np.zeros((T, N),            dtype=np.float32)
        self.old_logits = np.zeros((T, N, act_dim),   dtype=np.float32)
        # Filled after rollout
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns    = np.zeros((T, N), dtype=np.float32)

    # ── step-by-step filling ──────────────────────────

    def store(self, t, obs, action, logprob, reward, done, value, logits):
        self.obs[t]        = obs
        self.actions[t]    = action
        self.logprobs[t]   = logprob
        self.rewards[t]    = reward
        self.dones[t]      = done
        self.values[t]     = value
        self.old_logits[t] = logits

    # ── GAE computation ───────────────────────────────

    def compute_gae(self, next_value: np.ndarray, next_done: np.ndarray,
                    gamma: float, lam: float) -> None:
        """
        Standard GAE-λ (reverse sweep).
        δ_t = r_t + γ·V(s_{t+1})·(1-d_{t+1}) − V(s_t)
        A_t = Σ (γλ)^l · δ_{t+l}
        """
        gae = np.zeros(self.N, dtype=np.float32)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                nv  = next_value
                nnt = 1.0 - next_done
            else:
                nv  = self.values[t + 1]
                nnt = 1.0 - self.dones[t + 1]
            delta = self.rewards[t] + gamma * nv * nnt - self.values[t]
            gae   = delta + gamma * lam * nnt * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    # ── minibatch iterator ────────────────────────────

    def iter_minibatches(self, mb_size: int):
        """Flatten [T,N] → [T*N], shuffle, yield MiniBatch objects."""
        def flat(arr):
            return torch.tensor(
                arr.reshape(-1, *arr.shape[2:]),   # keep trailing dims
                device=self.device,
            )

        b_obs    = flat(self.obs)
        b_act    = flat(self.actions)
        b_logp   = flat(self.logprobs)
        b_adv    = flat(self.advantages)
        b_ret    = flat(self.returns)
        b_val    = flat(self.values)
        b_logits = flat(self.old_logits)

        total = b_obs.shape[0]
        idx   = np.random.permutation(total)

        for start in range(0, total, mb_size):
            mb = idx[start : start + mb_size]
            yield MiniBatch(
                obs        = b_obs[mb],
                actions    = b_act[mb],
                logprobs   = b_logp[mb],
                advantages = b_adv[mb],
                returns    = b_ret[mb],
                values     = b_val[mb],
                old_logits = b_logits[mb],
            )
