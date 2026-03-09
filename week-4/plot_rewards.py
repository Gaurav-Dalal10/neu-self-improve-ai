import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REWARD_PATTERN = re.compile(r"Reward:\s*([0-9]+(?:\.[0-9]+)?)")


def load_rewards(log_path: Path) -> np.ndarray:
    rewards = []

    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"]
    last_error = None

    for encoding in encodings:
        try:
            with log_path.open("r", encoding=encoding) as file:
                for line in file:
                    match = REWARD_PATTERN.search(line)
                    if match:
                        rewards.append(float(match.group(1)))
            if rewards:
                break
        except UnicodeDecodeError as error:
            rewards.clear()
            last_error = error

    if not rewards and last_error is not None:
        raise ValueError(f"Unable to decode log file with common encodings: {log_path}") from last_error

    if not rewards:
        raise ValueError(f"No rewards found in log file: {log_path}")

    return np.array(rewards, dtype=np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values

    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float32) / window
    averaged = np.convolve(values, kernel, mode="valid")

    prefix = np.full(window - 1, np.nan, dtype=np.float32)
    return np.concatenate([prefix, averaged])


def plot_rewards(rewards: np.ndarray, output_path: Path, window: int) -> None:
    episodes = np.arange(len(rewards))
    ma = moving_average(rewards, window)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Episode reward", alpha=0.45)
    plt.plot(episodes, ma, label=f"Moving average ({window})", linewidth=2)
    plt.title("REINFORCE on CartPole-v1")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CartPole training rewards from log file.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("training_log.txt"),
        help="Path to training log file (default: training_log.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reward_curve.png"),
        help="Path to output image file (default: reward_curve.png)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Moving-average window size (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rewards = load_rewards(args.log_file)
    plot_rewards(rewards, args.output, args.window)

    print(f"Saved reward plot to: {args.output}")


if __name__ == "__main__":
    main()
