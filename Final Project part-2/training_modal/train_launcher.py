import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional


def _require_endpoint_env() -> None:
    missing = []
    if not os.getenv("OPENAI_COMPAT_BASE_URL"):
        missing.append("OPENAI_COMPAT_BASE_URL")
    if not os.getenv("OPENAI_COMPAT_API_KEY"):
        missing.append("OPENAI_COMPAT_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required endpoint env vars: {', '.join(missing)}")


def _heartbeat(last_print: float, heartbeat_sec: int, payload: Dict) -> float:
    now = time.time()
    if now - last_print >= heartbeat_sec:
        print("[heartbeat] " + json.dumps(payload, ensure_ascii=True))
        return now
    return last_print


def _run_mock_loop(args: argparse.Namespace, metrics_path: Path) -> Dict:
    start = time.time()
    last_print = start
    best_reward = float("-inf")
    best_at = start
    stop_reason = "completed"

    for step in range(1, args.max_steps + 1):
        elapsed = time.time() - start
        if elapsed > args.max_wall_seconds:
            stop_reason = "max_wall_seconds_reached"
            break

        # Placeholder trajectory-level metrics; swap with real training logs in backend=command mode.
        reward = 0.0
        loss = 0.0
        row = {
            "step": step,
            "reward": reward,
            "loss": loss,
            "mode": args.mode,
            "elapsed_sec": round(elapsed, 2),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if reward > best_reward:
            best_reward = reward
            best_at = time.time()

        stagnation_sec = time.time() - best_at
        if stagnation_sec > args.stagnation_patience_minutes * 60:
            stop_reason = "stagnation_early_stop"
            break

        last_print = _heartbeat(
            last_print,
            args.heartbeat_sec,
            {
                "mode": args.mode,
                "step": step,
                "elapsed_sec": round(elapsed, 2),
                "best_reward": best_reward,
                "stagnation_sec": round(stagnation_sec, 2),
            },
        )
        time.sleep(0.05)

    return {
        "stop_reason": stop_reason,
        "best_reward": best_reward if best_reward != float("-inf") else None,
        "elapsed_sec": round(time.time() - start, 2),
    }


def _run_command_backend(args: argparse.Namespace, metrics_path: Path) -> Dict:
    if not args.command.strip():
        raise ValueError("backend=command requires --command")

    start = time.time()
    proc = subprocess.Popen(shlex.split(args.command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    last_print = start
    stop_reason = "completed"
    while True:
        elapsed = time.time() - start
        if elapsed > args.max_wall_seconds:
            proc.terminate()
            stop_reason = "max_wall_seconds_reached"
            break

        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"log": line.rstrip(), "elapsed_sec": round(elapsed, 2)}) + "\n")

        last_print = _heartbeat(
            last_print,
            args.heartbeat_sec,
            {
                "mode": args.mode,
                "elapsed_sec": round(elapsed, 2),
                "backend": "command",
                "pid": proc.pid,
            },
        )

        if proc.poll() is not None:
            if proc.returncode != 0:
                stop_reason = f"command_failed_rc_{proc.returncode}"
            break

        time.sleep(0.2)

    return {
        "stop_reason": stop_reason,
        "elapsed_sec": round(time.time() - start, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="final_project_part2/training_modal/checkpoints")
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--max_wall_seconds", type=int, default=600)
    p.add_argument("--mode", type=str, default="dev")
    p.add_argument("--heartbeat_sec", type=int, default=60)
    p.add_argument("--stagnation_patience_minutes", type=int, default=3)
    p.add_argument("--backend", choices=["mock", "command"], default="mock")
    p.add_argument("--command", type=str, default="")
    p.add_argument("--require_endpoint", action="store_true")
    args = p.parse_args()

    if args.mode == "dev" and args.max_wall_seconds > 600:
        raise ValueError("Dev runs must be <= 600 seconds")
    if args.mode == "final" and args.max_wall_seconds > 21600:
        raise ValueError("Final runs must be <= 21600 seconds (6h)")

    if args.require_endpoint:
        _require_endpoint_env()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / f"metrics_{args.mode}.jsonl"

    print(
        "[launch] "
        + json.dumps(
            {
                "mode": args.mode,
                "backend": args.backend,
                "max_steps": args.max_steps,
                "max_wall_seconds": args.max_wall_seconds,
                "output_dir": str(out),
            }
        )
    )

    summary = _run_mock_loop(args, metrics_path) if args.backend == "mock" else _run_command_backend(args, metrics_path)

    summary_path = out / f"run_summary_{args.mode}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
