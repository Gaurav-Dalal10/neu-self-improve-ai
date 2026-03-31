# Reproducibility Guide

## Environment

- Python 3.10+
- PyTorch 2.0+
- MiniGrid 2.3+
- OS: tested on Ubuntu 22.04 and macOS 14

## Install

```bash
pip install -r requirements.txt
```

## Reproduce Table 1 (Four Rooms)

Run all three algorithms across 3 seeds, matching the paper's evaluation protocol.

```bash
# Seed 0
python -m rl.train --algo reinforce --seed 0 --total_updates 300
python -m rl.train --algo ppo       --seed 0 --total_updates 300
python -m rl.train --algo polyppo   --seed 0 --total_updates 300

# Seed 1
python -m rl.train --algo reinforce --seed 1 --total_updates 300
python -m rl.train --algo ppo       --seed 1 --total_updates 300
python -m rl.train --algo polyppo   --seed 1 --total_updates 300

# Seed 2
python -m rl.train --algo reinforce --seed 2 --total_updates 300
python -m rl.train --algo ppo       --seed 2 --total_updates 300
python -m rl.train --algo polyppo   --seed 2 --total_updates 300
```

Each run writes `results.json` with `mean_return` and `mean_success`.
Average across the 3 seeds to compare against Table 1.

## Expected Runtime

| Algorithm  | ~Time (CPU, 300 updates) |
|-----------|-------------------------|
| REINFORCE | 10–20 min               |
| PPO       | 15–30 min               |
| Poly-PPO  | 30–60 min               |

Poly-PPO is slower because vine sampling runs N=8 extra rollouts
at p=2 states per update iteration.

## Run Tests

```bash
pytest tests/ -v
```

All 8 tests should pass. The smoke tests take ~30 seconds each
(they run 2 training updates with small configs).

## Hyperparameters

Defaults match the paper (Table 3) where specified.
Key parameters that may need tuning:

- `--beta_kl` : KL penalty coefficient. Paper sweeps {0.005, 0.01, 0.05, 0.1}.
  Default is 0.01.
- `--gamma` : Discount factor. Paper uses 1.0 for Four Rooms but
  0.99 often works better when training from scratch.
- `--total_updates` : More updates may improve results. Paper does not
  specify exact count.

## GPU

Add `--device cuda` to use GPU. All three algorithms support it.
