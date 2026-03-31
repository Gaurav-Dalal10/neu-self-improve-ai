"""
Proximal Policy Optimisation (PPO) with clipped surrogate objective.

Follows the standard CleanRL-style loop:
  1.  Collect T steps across N parallel envs  (no grad)
  2.  Compute GAE advantages                  (no grad)
  3.  K epochs of minibatch updates            (with grad)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence

from .buffer import RolloutBuffer
from .envs import make_vec_env
from .logging import CSVLogger
from .networks import ActorCritic
from .utils import explained_variance, to_numpy


@dataclass
class PPOConfig:
    env_id: str = "MiniGrid-FourRooms-v0"
    seed: int = 0
    total_updates: int = 300
    num_envs: int = 4
    num_steps: int = 128          # T – rollout length per env
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    anneal_lr: bool = True
    clip_eps: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4           # K
    minibatch_size: int = 64
    hidden_dim: int = 128
    device: str = "cpu"
    beta_kl: float = 0.0         # optional KL penalty
    pretrained_path: str = ""    # path to pretrained weights (empty = train from scratch)


def ppo_train(cfg: PPOConfig, run_dir: str, logger: CSVLogger,
              poly_bonus: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Main PPO training loop.

    If *poly_bonus* is not None it is an [T, N] array added to advantages
    after GAE – this hook lets Poly-PPO reuse the same code path.
    """
    device = torch.device(cfg.device)
    envs = make_vec_env(cfg.env_id, seed=cfg.seed, n=cfg.num_envs)

    obs, _ = envs.reset(seed=cfg.seed)
    obs_dim = int(obs.shape[-1])
    act_dim = int(envs.single_action_space.n)

    model = ActorCritic(obs_dim, act_dim, cfg.hidden_dim).to(device)
    if cfg.pretrained_path:
        model.load_state_dict(torch.load(cfg.pretrained_path, map_location=device))
        print(f"  Loaded pretrained weights from {cfg.pretrained_path}")
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    buf = RolloutBuffer(cfg.num_steps, cfg.num_envs, obs_dim, act_dim, device)

    ep_returns: List[float] = []
    latest: Dict[str, float] = {}
    global_step = 0

    for update in range(1, cfg.total_updates + 1):

        # ── LR annealing ──
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / max(cfg.total_updates, 1)
            optim.param_groups[0]["lr"] = frac * cfg.lr

        # ════════════════════════════════════
        # PHASE 1 – rollout collection
        # ════════════════════════════════════
        for t in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logprob, value, logits = model.act(obs_t)

            nxt, reward, term, trunc, infos = envs.step(to_numpy(action))
            done = np.logical_or(term, trunc).astype(np.float32)

            buf.store(t, obs, to_numpy(action), to_numpy(logprob),
                      reward, done, to_numpy(value), to_numpy(logits))
            obs = nxt

            # track episode returns
            ep = infos.get("episode")
            mask = infos.get("_episode")
            if ep is not None and mask is not None:
                for i, flag in enumerate(mask):
                    if flag:
                        ep_returns.append(float(ep["r"][i]))
            fi = infos.get("final_info")
            if fi is not None:
                for item in fi:
                    if item and "episode" in item:
                        ep_returns.append(float(item["episode"]["r"]))

        # ════════════════════════════════════
        # PHASE 2 – GAE
        # ════════════════════════════════════
        with torch.no_grad():
            nxt_t = torch.tensor(obs, dtype=torch.float32, device=device)
            _, nxt_val, _ = model.get_dist_value(nxt_t)
        buf.compute_gae(to_numpy(nxt_val), buf.dones[-1],
                        gamma=cfg.gamma, lam=cfg.gae_lambda)

        # optional poly bonus (used by PolyPPO)
        if poly_bonus is not None:
            buf.advantages += poly_bonus
            buf.returns = buf.advantages + buf.values

        # normalise advantages globally
        flat_adv = buf.advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        buf.advantages = flat_adv.reshape(cfg.num_steps, cfg.num_envs)

        # ════════════════════════════════════
        # PHASE 3 – PPO update epochs
        # ════════════════════════════════════
        pg_losses, v_losses, entropies, kl_vals, clip_fracs = [], [], [], [], []

        for _ in range(cfg.ppo_epochs):
            for mb in buf.iter_minibatches(cfg.minibatch_size):
                dist, new_val, _ = model.get_dist_value(mb.obs)
                new_logp = dist.log_prob(mb.actions)
                ent      = dist.entropy().mean()

                log_ratio = new_logp - mb.logprobs
                ratio     = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    kl_vals.append(float(approx_kl))
                    clip_fracs.append(
                        float(((ratio - 1).abs() > cfg.clip_eps).float().mean()))

                # clipped policy loss
                pg1 = -mb.advantages * ratio
                pg2 = -mb.advantages * torch.clamp(ratio,
                                                   1 - cfg.clip_eps,
                                                   1 + cfg.clip_eps)
                pg_loss = torch.max(pg1, pg2).mean()

                # clipped value loss
                if cfg.clip_vloss:
                    vu = (new_val - mb.returns) ** 2
                    vc = (mb.values + torch.clamp(new_val - mb.values,
                                                  -cfg.clip_eps, cfg.clip_eps)
                          - mb.returns) ** 2
                    v_loss = 0.5 * torch.max(vu, vc).mean()
                else:
                    v_loss = 0.5 * F.mse_loss(new_val, mb.returns)

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                # optional KL penalty (used by PolyPPO)
                if cfg.beta_kl > 0:
                    old_dist = Categorical(logits=mb.old_logits)
                    kl = kl_divergence(old_dist, dist).mean()
                    loss = loss + cfg.beta_kl * kl

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.max_grad_norm)
                optim.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(ent.item()))

        # ── logging ──
        latest = {
            "update": update,
            "global_step": global_step,
            "charts/episodic_return":
                float(np.mean(ep_returns[-20:])) if ep_returns else np.nan,
            "losses/policy_loss":  float(np.mean(pg_losses)),
            "losses/value_loss":   float(np.mean(v_losses)),
            "losses/entropy":      float(np.mean(entropies)),
            "losses/approx_kl":    float(np.mean(kl_vals)) if kl_vals else 0,
            "losses/clipfrac":     float(np.mean(clip_fracs)) if clip_fracs else 0,
            "losses/explained_var": explained_variance(
                buf.values.reshape(-1), buf.returns.reshape(-1)),
            "lr": float(optim.param_groups[0]["lr"]),
        }
        logger.log(latest)

    torch.save(model.state_dict(), f"{run_dir}/model.pt")
    envs.close()
    return latest
