"""
Evaluate a saved model over many episodes and write results.json.

Paper protocol: 100 rollouts across 50 configurations.
We evaluate on seeds 0-49 (the same fixed configs used in training).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
from torch.distributions import Categorical

from .envs import make_env
from .networks import ActorCritic, Actor


def evaluate_run(run_dir: str, episodes: int = 100,
                 device: str = "cpu") -> Dict[str, float]:
    """Load config + model from *run_dir*, evaluate on 50 configs, save results."""
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    algo   = cfg["algo"]
    env_id = cfg["env_id"]
    hidden = int(cfg.get("hidden_dim", 128))

    # probe obs/act dims
    tmp = make_env(env_id, seed=0)
    obs_tmp, _ = tmp.reset(seed=0)
    obs_dim = int(obs_tmp.shape[-1])
    act_dim = int(tmp.action_space.n)
    tmp.close()

    # load the right architecture
    if algo in ("ppo", "polyppo"):
        model = ActorCritic(obs_dim, act_dim, hidden).to(device)
        model.load_state_dict(
            torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
        model.eval()
    else:
        model = Actor(obs_dim, act_dim, hidden).to(device)
        state = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
        model.load_state_dict(state["actor"])
        model.eval()

    # ── Evaluate: 50 configs × rollouts_per config ──
    num_configs = 50
    rollouts_per = max(1, episodes // num_configs)
    returns, successes = [], []

    for cfg_seed in range(num_configs):
        env = make_env(env_id, seed=cfg_seed)
        for _ in range(rollouts_per):
            obs, _ = env.reset(seed=cfg_seed)
            done = trunc = False
            total_r = 0.0
            while not (done or trunc):
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                with torch.no_grad():
                    if algo in ("ppo", "polyppo"):
                        dist, _, _ = model.get_dist_value(obs_t)
                    else:
                        dist = Categorical(logits=model(obs_t))
                    action = dist.sample().item()
                obs, reward, done, trunc, info = env.step(action)
                total_r += float(reward)

            returns.append(total_r)
            successes.append(int(total_r > 0))
        env.close()

    result = {
        "mean_return":  float(np.mean(returns)),
        "std_return":   float(np.std(returns)),
        "mean_success": float(np.mean(successes)),
        "episodes":     len(returns),
    }
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    r = evaluate_run(args.run_dir, args.episodes, args.device)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
