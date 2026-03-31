"""Smoke tests: each algorithm runs 2 updates and produces artefacts."""
import argparse
import os
from pathlib import Path

from rl.train import run_once


def _make_args(algo: str, tmp: Path) -> argparse.Namespace:
    return argparse.Namespace(
        algo=algo,
        env="MiniGrid-FourRooms-v0",
        seed=0,
        total_updates=2,
        output_dir=str(tmp / "outputs"),
        device="cpu",
        num_envs=2,
        num_steps=32,
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        lr_value=1e-3,
        clip_eps=0.2,
        ppo_epochs=2,
        minibatch_size=32,
        hidden_dim=64,
        beta_kl=0.01,
        episodes_per_update=2,
        poly_N=4,
        poly_n=2,
        poly_M=2,
        poly_p=1,
        poly_W=3,
        eval_episodes=5,
         pretrained_path="",
    )


def _check(algo: str, tmp: Path):
    d = run_once(_make_args(algo, tmp))
    assert os.path.exists(os.path.join(d, "metrics.csv"))
    assert os.path.exists(os.path.join(d, "model.pt"))
    assert os.path.exists(os.path.join(d, "results.json"))


def test_smoke_ppo(tmp_path: Path):
    _check("ppo", tmp_path)


def test_smoke_reinforce(tmp_path: Path):
    _check("reinforce", tmp_path)


def test_smoke_polyppo(tmp_path: Path):
    _check("polyppo", tmp_path)
