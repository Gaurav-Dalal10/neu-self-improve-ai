# REINFORCE on CartPole-v1

A minimal implementation of the **REINFORCE (Monte Carlo Policy Gradient)** algorithm using PyTorch and Gymnasium on `CartPole-v1`.

## Project Structure

- `reinforce_cartpole.py` — training script
- `plot_rewards.py` — reward-curve plotting utility
- `requirements.txt` — Python dependencies
- `training_log.txt` — sample training output (episode rewards)
- `reward_curve.png` — generated training curve image

## Requirements

- Python 3.9+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

From the project folder:

```bash
python reinforce_cartpole.py
```

The script trains for **1000 episodes** and prints reward per episode in this format:

```text
Episode 123 | Reward: 45.0
```

## Recorded Results (Current Run)

Results below are from the latest run captured in `training_log.txt` (1000 episodes):

- Episodes: **1000**
- Average reward (first 10 episodes): **34.2**
- Average reward (last 10 episodes): **142.9**
- Average reward (last 100 episodes): **132.8**
- Maximum episode reward: **400**
- Minimum episode reward: **9**
- Final episode reward: **79**

These numbers show learning progress over time (early performance is low, later averages are much higher).

## Notes

- Environment: `CartPole-v1`
- Optimizer: Adam (`lr=2e-4`)
- Discount factor: `gamma=0.99`
- Policy network: 2-layer MLP (`obs_dim -> 128 -> act_dim`) with softmax output

## Quick Reproduce of Results Log

To save output to a file:

```bash
python reinforce_cartpole.py > training_log.txt
```

You can rerun multiple times to compare learning stability across seeds/runs.

## Plot Training Curve

Generate a reward plot from the saved log:

```bash
python plot_rewards.py --log-file training_log.txt --output reward_curve.png --window 50
```

Arguments:

- `--log-file`: input training log path
- `--output`: output image path
- `--window`: moving average window size

The script creates `reward_curve.png` in the project folder by default.
