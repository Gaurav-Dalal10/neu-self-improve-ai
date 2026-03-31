"""
Single entry-point for all three algorithms.

Usage:
  python -m rl.pretrain                                  # step 0: pretrain
  python -m rl.train --algo reinforce --pretrained_path pretrained.pt
  python -m rl.train --algo ppo       --pretrained_path pretrained.pt
  python -m rl.train --algo polyppo   --pretrained_path pretrained.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .evaluate import evaluate_run
from .logging import CSVLogger, plot_curve
from .polyppo import PolyPPOConfig, polyppo_train
from .ppo import PPOConfig, ppo_train
from .reinforce import ReinforceConfig, reinforce_train
from .utils import ensure_dir, set_seed


def _run_dir(base: str, env_id: str, algo: str, seed: int) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(base, env_id.replace("/", "_"), algo, f"seed{seed}_{ts}")
    ensure_dir(d)
    return d


def run_once(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    run_dir = _run_dir(args.output_dir, args.env, args.algo, args.seed)
    logger = CSVLogger(os.path.join(run_dir, "metrics.csv"))

    if args.algo == "reinforce":
        cfg = ReinforceConfig(
            env_id=args.env, seed=args.seed,
            total_updates=args.total_updates,
            episodes_per_update=args.episodes_per_update,
            gamma=args.gamma, lr_actor=args.lr, lr_critic=args.lr_value,
            hidden_dim=args.hidden_dim, device=args.device,
            pretrained_path=args.pretrained_path,
        )
        _save(run_dir, "reinforce", cfg)
        reinforce_train(cfg, run_dir, logger)

    elif args.algo == "ppo":
        cfg = PPOConfig(
            env_id=args.env, seed=args.seed,
            total_updates=args.total_updates,
            num_envs=args.num_envs, num_steps=args.num_steps,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            lr=args.lr, clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            hidden_dim=args.hidden_dim, device=args.device,
            beta_kl=args.beta_kl,
            pretrained_path=args.pretrained_path,
        )
        _save(run_dir, "ppo", cfg)
        ppo_train(cfg, run_dir, logger)

    elif args.algo == "polyppo":
        cfg = PolyPPOConfig(
            env_id=args.env, seed=args.seed,
            total_updates=args.total_updates,
            num_envs=args.num_envs, num_steps=args.num_steps,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            lr=args.lr, clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            hidden_dim=args.hidden_dim, device=args.device,
            beta_kl=args.beta_kl,
            pretrained_path=args.pretrained_path,
            vine_N=args.poly_N, set_n=args.poly_n,
            num_sets_M=args.poly_M, num_rollout_p=args.poly_p,
            window_W=args.poly_W,
        )
        _save(run_dir, "polyppo", cfg)
        polyppo_train(cfg, run_dir, logger)

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    logger.flush()
    plot_curve(os.path.join(run_dir, "metrics.csv"),
               os.path.join(run_dir, "learning_curve.png"))
    result = evaluate_run(run_dir, episodes=args.eval_episodes,
                          device=args.device)
    print(f"\n{'='*50}")
    print(f"  {args.algo.upper()} | seed={args.seed}")
    print(f"  mean_return  = {result['mean_return']:.3f}")
    print(f"  success_rate = {result['mean_success']*100:.1f}%")
    print(f"  run_dir      = {run_dir}")
    print(f"{'='*50}\n")
    return run_dir


def _save(run_dir: str, algo: str, cfg) -> None:
    d = {"algo": algo, **asdict(cfg)}
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL on MiniGrid Four Rooms")
    p.add_argument("--algo",  required=True,
                   choices=["ppo", "reinforce", "polyppo"])
    p.add_argument("--env",   default="MiniGrid-FourRooms-v0")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--total_updates", type=int, default=300)
    p.add_argument("--output_dir",    default="outputs")
    p.add_argument("--device",        default="cpu")
    p.add_argument("--pretrained_path", type=str, default="",
                   help="Path to pretrained.pt (from rl.pretrain)")

    # PPO / PolyPPO
    p.add_argument("--num_envs",      type=int,   default=4)
    p.add_argument("--num_steps",     type=int,   default=128)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae_lambda",    type=float, default=0.95)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--lr_value",      type=float, default=1e-3)
    p.add_argument("--clip_eps",      type=float, default=0.2)
    p.add_argument("--ppo_epochs",    type=int,   default=4)
    p.add_argument("--minibatch_size",type=int,   default=64)
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--beta_kl",       type=float, default=0.01)

    # REINFORCE
    p.add_argument("--episodes_per_update", type=int, default=8)

    # PolyPPO
    p.add_argument("--poly_N", type=int, default=8)
    p.add_argument("--poly_n", type=int, default=4)
    p.add_argument("--poly_M", type=int, default=4)
    p.add_argument("--poly_p", type=int, default=2)
    p.add_argument("--poly_W", type=int, default=5)

    # Eval
    p.add_argument("--eval_episodes", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_once(args)


if __name__ == "__main__":
    main()
