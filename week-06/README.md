# Polychromic PPO — Four Rooms Replication

Replicates Table 1 results for **MiniGrid-FourRooms-v0** from:

> *Polychromic Objectives for Reinforcement Learning*
> Hamid, Orney, Xu, Finn, Sadigh (Stanford, 2026)

## Table 1 Targets (Four Rooms, no UCB)

| Method       | Avg Reward | Success % |
|-------------|-----------|----------|
| REINFORCE   | 0.639     | 89.6     |
| PPO         | 0.618     | 89.2     |
| **Poly-PPO** | **0.666** | **92.4** |

## Quick Start

```bash
pip install -r requirements.txt

# Train each algorithm (from project root)
python -m rl.train --algo reinforce
python -m rl.train --algo ppo
python -m rl.train --algo polyppo

# Run unit tests
pytest tests/ -v
```

Each run saves to `outputs/<env>/<algo>/seed<N>_<timestamp>/` containing:
- `model.pt` — trained weights
- `metrics.csv` — per-update loss / return logs
- `learning_curve.png` — reward vs updates plot
- `results.json` — final evaluation (mean return + success rate)
- `config.json` — full hyperparameter dump

## Run With Custom Hyperparameters

```bash
python -m rl.train --algo polyppo --seed 42 --total_updates 500 \
    --num_envs 8 --lr 1e-3 --gamma 0.97 --beta_kl 0.05 --device cuda
```

## Project Structure

```
rl/
  buffer.py      RolloutBuffer [T,N] with GAE and minibatch iterator
  envs.py        Flat-obs wrapper, SyncVectorEnv, state save/restore, diversity
  networks.py    Actor, Critic, ActorCritic MLPs with orthogonal init
  reinforce.py   REINFORCE with learned value baseline
  ppo.py         PPO — clipped surrogate, GAE, vectorised envs
  polyppo.py     Poly-PPO — vine sampling + polychromic advantage bonus
  evaluate.py    Load saved model, run episodes, write results.json
  logging.py     CSVLogger + matplotlib learning curve
  train.py       Single CLI entry point dispatching all 3 algorithms
  utils.py       set_seed, to_numpy, explained_variance

tests/
  test_gae.py          GAE matches hand-computed reference
  test_clip.py         PPO clip selects pessimistic branch
  test_diversity.py    Room-set diversity metric bounds
  test_env_restore.py  MiniGrid state save → restore round-trip
  test_smoke.py        Each algo trains 2 updates without crashing
```

## How the Algorithms Work

**REINFORCE** — collect full episodes, compute G_t, advantage = G_t − V(s_t),
single gradient step per update batch.

**PPO** — collect T=128 steps across N=4 parallel envs, compute GAE-λ,
K=4 epochs of clipped minibatch updates.

**Poly-PPO** — same PPO loop plus: (1) save env-0 state at each timestep,
(2) after rollout, pick p=2 rollout states, branch N=8 vine trajectories
from each via state restore, (3) form M=4 sets of n=4 trajectories,
score each set as avg_reward × diversity, (4) add the bonus to GAE
advantages in a window of W=5 steps. KL penalty keeps the policy stable.
