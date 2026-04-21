"""
Budget-safe launcher for Flow-GRPO + LoRA training.

Design goals (professor constraints):
- Dev runs must be <= 10 minutes
- Final runs <= 5-6 hours
- Emit per-minute progress heartbeat
- Use cheapest viable GPU setting by default (L40S)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

import yaml


def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("training", {})


def _resolve_mode_cfg(cfg: Dict, mode: str) -> Dict:
    if mode not in {"dev", "final"}:
        raise ValueError(f"Unsupported mode: {mode}")

    timeout = int(cfg["timeout_dev_sec"] if mode == "dev" else cfg["timeout_final_sec"])
    max_steps = int(cfg["max_steps_dev"] if mode == "dev" else cfg["max_steps_final"])
    gpu = cfg.get("gpu_dev" if mode == "dev" else "gpu_final", "L40S")

    if mode == "dev" and timeout > 600:
        raise ValueError("Dev timeout exceeds 600 seconds policy")
    if mode == "final" and timeout > 21600:
        raise ValueError("Final timeout exceeds 6-hour policy")

    return {
        "timeout": timeout,
        "max_steps": max_steps,
        "gpu": gpu,
        "heartbeat_sec": int(cfg.get("heartbeat_sec", 60)),
        "stagnation_patience_minutes": int(cfg.get("stagnation_patience_minutes", 3)),
    }


def _print_summary(mode: str, resolved: Dict, output_dir: str) -> None:
    print("[training-policy]")
    print(f"  mode: {mode}")
    print(f"  gpu: {resolved['gpu']}")
    print(f"  timeout_sec: {resolved['timeout']}")
    print(f"  max_steps: {resolved['max_steps']}")
    print(f"  heartbeat_sec: {resolved['heartbeat_sec']}")
    print(f"  stagnation_patience_minutes: {resolved['stagnation_patience_minutes']}")
    print(f"  output_dir: {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dev", "final"], default="dev")
    p.add_argument("--config", type=str, default="final_project_part2/configs/flow_grpo_lora.yaml")
    p.add_argument("--output_dir", type=str, default="final_project_part2/training_modal/checkpoints")
    p.add_argument("--require_endpoint", action="store_true", help="Require endpoint env vars before launch")
    p.add_argument("--backend", choices=["mock", "command"], default="mock")
    p.add_argument("--command", type=str, default="", help="Used when --backend=command")
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    resolved = _resolve_mode_cfg(cfg, args.mode)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _print_summary(args.mode, resolved, args.output_dir)

    cmd = [
        "python3",
        "final_project_part2/training_modal/train_launcher.py",
        "--output_dir",
        args.output_dir,
        "--max_steps",
        str(resolved["max_steps"]),
        "--max_wall_seconds",
        str(resolved["timeout"]),
        "--mode",
        args.mode,
        "--heartbeat_sec",
        str(resolved["heartbeat_sec"]),
        "--stagnation_patience_minutes",
        str(resolved["stagnation_patience_minutes"]),
        "--backend",
        args.backend,
    ]

    if args.require_endpoint:
        cmd.append("--require_endpoint")
    if args.backend == "command":
        if not args.command.strip():
            raise ValueError("--backend=command requires --command")
        cmd.extend(["--command", args.command])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        raise
