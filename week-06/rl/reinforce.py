"""
REINFORCE with a learned value baseline.

Collects full episodes, computes returns-to-go, uses V(s) as baseline:
    ∇J ≈  Σ_t  (G_t − V(s_t)) · ∇ log π(a_t|s_t)

Separate actor and critic optimisers (no shared backbone needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .envs import make_env
from .logging import CSVLogger
from .networks import Actor, Critic
from .utils import to_numpy


@dataclass
class ReinforceConfig:
    env_id: str = "MiniGrid-FourRooms-v0"
    seed: int = 0
    total_updates: int = 300
    episodes_per_update: int = 8
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    device: str = "cpu"
    pretrained_path: str = ""    # path to pretrained ActorCritic weights


def _returns_to_go(rewards: List[float], gamma: float) -> np.ndarray:
    """Compute discounted returns G_t = Σ γ^k r_{t+k}."""
    G = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def reinforce_train(cfg: ReinforceConfig, run_dir: str,
                    logger: CSVLogger) -> Dict[str, float]:
    device = torch.device(cfg.device)
    env = make_env(cfg.env_id, cfg.seed)
    obs, _ = env.reset(seed=cfg.seed)

    obs_dim = int(obs.shape[-1])
    act_dim = int(env.action_space.n)

    actor  = Actor(obs_dim, act_dim, cfg.hidden_dim).to(device)
    critic = Critic(obs_dim, cfg.hidden_dim).to(device)

    if cfg.pretrained_path:
        # pretrained.pt is an ActorCritic; extract actor.* and critic.* keys
        sd = torch.load(cfg.pretrained_path, map_location=device)
        actor.load_state_dict(
            {k.removeprefix("actor."): v for k, v in sd.items() if k.startswith("actor.")})
        critic.load_state_dict(
            {k.removeprefix("critic."): v for k, v in sd.items() if k.startswith("critic.")})
        print(f"  Loaded pretrained weights from {cfg.pretrained_path}")

    opt_a = torch.optim.Adam(actor.parameters(),  lr=cfg.lr_actor,  eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic, eps=1e-5)

    recent_returns: List[float] = []
    latest: Dict[str, float] = {}

    for update in range(1, cfg.total_updates + 1):
        all_obs, all_act, all_ret, all_adv = [], [], [], []

        for _ in range(cfg.episodes_per_update):
            obs, _ = env.reset()
            ep_obs, ep_act, ep_rew = [], [], []
            done = trunc = False

            while not (done or trunc):
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                logits = actor(obs_t)
                action = Categorical(logits=logits).sample()
                nxt, reward, done, trunc, _ = env.step(int(action.item()))
                ep_obs.append(obs)
                ep_act.append(int(action.item()))
                ep_rew.append(float(reward))
                obs = nxt

            G = _returns_to_go(ep_rew, cfg.gamma)
            with torch.no_grad():
                vals = critic(
                    torch.tensor(np.array(ep_obs), dtype=torch.float32,
                                 device=device)
                ).cpu().numpy()
            adv = G - vals

            all_obs.extend(ep_obs)
            all_act.extend(ep_act)
            all_ret.extend(G.tolist())
            all_adv.extend(adv.tolist())
            recent_returns.append(float(sum(ep_rew)))

        # ── flatten & normalise advantages ──
        obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(all_act), dtype=torch.int64,   device=device)
        ret_t = torch.tensor(np.array(all_ret), dtype=torch.float32, device=device)
        adv_t = torch.tensor(np.array(all_adv), dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── policy loss ──
        logits = actor(obs_t)
        logp   = Categorical(logits=logits).log_prob(act_t)
        pi_loss = -(logp * adv_t).mean()

        opt_a.zero_grad()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
        opt_a.step()

        # ── value loss ──
        v_loss = F.mse_loss(critic(obs_t), ret_t)

        opt_c.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        opt_c.step()

        # ── log ──
        latest = {
            "update": update,
            "charts/episodic_return": float(np.mean(recent_returns[-20:])),
            "losses/policy_loss":     float(pi_loss.item()),
            "losses/value_loss":      float(v_loss.item()),
        }
        logger.log(latest)

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
               f"{run_dir}/model.pt")
    env.close()
    return latest
