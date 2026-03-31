"""
MLP Actor-Critic for MiniGrid Four Rooms.

Architecture (matches PPO pseudocode):
  obs → Linear(hidden) → Tanh → Linear(hidden) → Tanh → head

Actor head:  → Linear(num_actions)  [logits, std=0.01]
Critic head: → Linear(1)           [value,  std=1.0]

Separate actor/critic networks (no shared backbone) so REINFORCE
can use just the actor while PPO/PolyPPO use both.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _ortho(layer: nn.Module, gain: float = np.sqrt(2), bias: float = 0.0):
    """Apply orthogonal init to a Linear layer."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, bias)


class Actor(nn.Module):
    """Policy network: obs → logits."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.apply(_ortho)
        _ortho(self.net[-1], gain=0.01)       # small init for logits

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Value network: obs → scalar V(s)."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.apply(_ortho)
        _ortho(self.net[-1], gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Bundles actor + critic for PPO / Poly-PPO."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.actor  = Actor(obs_dim, act_dim, hidden)
        self.critic = Critic(obs_dim, hidden)

    def get_dist_value(
        self, obs: torch.Tensor
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        return Categorical(logits=logits), self.critic(obs), logits

    def act(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, logits = self.get_dist_value(obs)
        action  = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value, logits
