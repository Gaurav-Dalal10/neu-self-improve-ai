"""
Polychromic PPO  (Algorithm 2 in the paper).

Builds on top of the standard PPO loop with two additions:

1. **Vine sampling**: after each rollout phase we pick p rollout states
   along env-0's trajectory, restore the env to each one, and branch
   N independent continuations.

2. **Polychromic advantage bonus**: for each rollout state we form M
   random sets of n trajectories, score each set with
       f_poly = mean_reward × diversity
   and add  (max_score − baseline)  to the GAE advantages in a window
   of W timesteps.  This additive bonus pushes the policy toward
   *diverse* strategies, not just high-reward ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence

from .buffer import RolloutBuffer
from .envs import (
    diversity, make_env, make_vec_env, obs_from_env,
    restore_state, rooms_visited, save_state,
)
from .logging import CSVLogger
from .networks import ActorCritic
from .utils import explained_variance, to_numpy


@dataclass
class PolyPPOConfig:
    # ── shared with PPO ──
    env_id: str = "MiniGrid-FourRooms-v0"
    seed: int = 0
    total_updates: int = 300
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    anneal_lr: bool = True
    clip_eps: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 64
    hidden_dim: int = 128
    device: str = "cpu"
    beta_kl: float = 0.01       # KL penalty for stability
    pretrained_path: str = ""   # path to pretrained weights

    # ── poly-specific ──
    vine_N: int = 8              # trajectories per rollout state
    set_n: int = 4               # trajectories per set
    num_sets_M: int = 4          # sets for baseline
    num_rollout_p: int = 2       # rollout states per trajectory
    window_W: int = 5            # advantage window
    max_vine_steps: int = 64     # max steps per vine trajectory
    poly_coef: float = 1.0       # scaling for the bonus


# ─────────────────────────────────────────────
#  Helper: normalise an array to [0, 1]
# ─────────────────────────────────────────────

def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


# ─────────────────────────────────────────────
#  Compute the polychromic bonus for one rollout
# ─────────────────────────────────────────────

def _poly_bonus(cfg: PolyPPOConfig, model: ActorCritic,
                saved_states: List[dict]) -> np.ndarray:
    """
    For each of p equally-spaced rollout states along env-0:
      1. Branch N vine trajectories
      2. Form M random sets of n trajectories
      3. Score each set  f_poly = avg_reward × diversity
      4. Advantage = max(scores) − mean(scores)
      5. Add to a window of W timesteps
    Returns: bonus array of shape [T, num_envs]
    """
    bonus = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.float32)
    if not saved_states:
        return bonus

    # pick p equally-spaced indices
    idxs = np.linspace(0, len(saved_states) - 1,
                       num=max(1, cfg.num_rollout_p), dtype=int)

    vine_env = make_env(cfg.env_id, seed=cfg.seed + 777)
    w = int(vine_env.unwrapped.width)
    h = int(vine_env.unwrapped.height)

    for t0 in idxs.tolist():
        snap = saved_states[t0]

        # ── collect N vine trajectories from this state ──
        vine_returns: List[float] = []
        vine_rooms:   List[frozenset] = []

        for _ in range(cfg.vine_N):
            vine_env.reset()
            restore_state(vine_env, snap)
            obs = obs_from_env(vine_env)
            total_r = 0.0
            positions = [tuple(int(x) for x in vine_env.unwrapped.agent_pos)]

            for _ in range(cfg.max_vine_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    dist, _, _ = model.get_dist_value(obs_t)
                    action = dist.sample().item()
                obs, r, done, trunc, _ = vine_env.step(action)
                total_r += float(r)
                positions.append(
                    tuple(int(x) for x in vine_env.unwrapped.agent_pos))
                if done or trunc:
                    break

            vine_returns.append(total_r)
            vine_rooms.append(rooms_visited(positions, w, h))

        ret_norm = _norm01(np.array(vine_returns, dtype=np.float32))

        # ── form M sets, score with f_poly ──
        scores: List[float] = []
        for _ in range(cfg.num_sets_M):
            replace = cfg.set_n > cfg.vine_N
            idxs_set = np.random.choice(cfg.vine_N, size=cfg.set_n,
                                        replace=replace)
            d = diversity([vine_rooms[i] for i in idxs_set], cfg.set_n)
            d_norm = np.clip(d, 0, 1)
            avg_r  = float(np.mean(ret_norm[idxs_set]))
            scores.append(avg_r * d_norm)

        baseline  = float(np.mean(scores))
        poly_adv  = max(scores) - baseline

        # ── apply to window ──
        end = min(cfg.num_steps, t0 + cfg.window_W)
        bonus[t0:end, 0] += cfg.poly_coef * poly_adv

    vine_env.close()
    return bonus


# ─────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────

def polyppo_train(cfg: PolyPPOConfig, run_dir: str,
                  logger: CSVLogger) -> Dict[str, float]:
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
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / max(cfg.total_updates, 1)
            optim.param_groups[0]["lr"] = frac * cfg.lr

        # ═══════════════════════════════════
        # PHASE 1 – rollout + save env-0 states
        # ═══════════════════════════════════
        saved_states: List[dict] = []

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

            # snapshot env-0 for vine sampling later
            saved_states.append(save_state(envs.envs[0]))

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

        # ═══════════════════════════════════
        # PHASE 2 – GAE + polychromic bonus
        # ═══════════════════════════════════
        with torch.no_grad():
            nxt_t = torch.tensor(obs, dtype=torch.float32, device=device)
            _, nxt_val, _ = model.get_dist_value(nxt_t)
        buf.compute_gae(to_numpy(nxt_val), buf.dones[-1],
                        gamma=cfg.gamma, lam=cfg.gae_lambda)

        poly = _poly_bonus(cfg, model, saved_states)
        buf.advantages += poly
        buf.returns = buf.advantages + buf.values

        flat_adv = buf.advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        buf.advantages = flat_adv.reshape(cfg.num_steps, cfg.num_envs)

        # ═══════════════════════════════════
        # PHASE 3 – PPO update epochs
        # ═══════════════════════════════════
        pg_losses, v_losses, entropies = [], [], []

        for _ in range(cfg.ppo_epochs):
            for mb in buf.iter_minibatches(cfg.minibatch_size):
                dist, new_val, _ = model.get_dist_value(mb.obs)
                new_logp = dist.log_prob(mb.actions)
                ent      = dist.entropy().mean()

                log_ratio = new_logp - mb.logprobs
                ratio     = log_ratio.exp()

                pg1 = -mb.advantages * ratio
                pg2 = -mb.advantages * torch.clamp(
                    ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                pg_loss = torch.max(pg1, pg2).mean()

                if cfg.clip_vloss:
                    vu = (new_val - mb.returns) ** 2
                    vc = (mb.values + torch.clamp(
                        new_val - mb.values, -cfg.clip_eps, cfg.clip_eps
                    ) - mb.returns) ** 2
                    v_loss = 0.5 * torch.max(vu, vc).mean()
                else:
                    v_loss = 0.5 * ((new_val - mb.returns) ** 2).mean()

                # KL penalty
                old_dist = Categorical(logits=mb.old_logits)
                kl = kl_divergence(old_dist, dist).mean()

                loss = (pg_loss
                        + cfg.vf_coef * v_loss
                        - cfg.ent_coef * ent
                        + cfg.beta_kl * kl)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.max_grad_norm)
                optim.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(ent.item()))

        latest = {
            "update": update,
            "global_step": global_step,
            "charts/episodic_return":
                float(np.mean(ep_returns[-20:])) if ep_returns else np.nan,
            "losses/policy_loss":   float(np.mean(pg_losses)),
            "losses/value_loss":    float(np.mean(v_losses)),
            "losses/entropy":       float(np.mean(entropies)),
            "losses/explained_var": explained_variance(
                buf.values.reshape(-1), buf.returns.reshape(-1)),
            "poly/bonus_mean":      float(np.mean(poly)),
            "lr": float(optim.param_groups[0]["lr"]),
        }
        logger.log(latest)

    torch.save(model.state_dict(), f"{run_dir}/model.pt")
    envs.close()
    return latest
