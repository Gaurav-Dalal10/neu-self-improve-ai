"""
Pretrain a policy on BFS-optimal expert demonstrations via behavioural cloning.

Flow:
  1. For each of 50 env seeds, run BFS on the full grid to find the shortest
     path from agent to goal.
  2. Convert paths to (flat_obs, action) pairs.
  3. Train an ActorCritic network with cross-entropy + entropy regulariser
     until it reaches ~70 % success (matching the paper's pretrained baseline).

Usage:
  python -m rl.pretrain                        # saves pretrained.pt
  python -m rl.pretrain --epochs 80 --out my_model.pt
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .envs import FlatObsWrapper, make_env
from .networks import ActorCritic

# ─────────────────────────────────────────────
#  MiniGrid action constants
# ─────────────────────────────────────────────
TURN_LEFT  = 0
TURN_RIGHT = 1
FORWARD    = 2

# direction index → (dx, dy)
_DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


# ─────────────────────────────────────────────
#  BFS on the full grid
# ─────────────────────────────────────────────

def _find_goal(env: gym.Env) -> Optional[Tuple[int, int]]:
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return (x, y)
    return None


def _bfs(env: gym.Env) -> Optional[List[Tuple[int, int]]]:
    """BFS from agent position to goal; returns list of (x,y) waypoints."""
    grid = env.unwrapped.grid
    start = tuple(int(c) for c in env.unwrapped.agent_pos)
    goal  = _find_goal(env)
    if goal is None:
        return None

    visited = {start}
    parent  = {start: None}
    q = deque([start])

    while q:
        pos = q.popleft()
        if pos == goal:
            path = []
            while pos is not None:
                path.append(pos)
                pos = parent[pos]
            return list(reversed(path))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (pos[0] + dx, pos[1] + dy)
            if nb in visited:
                continue
            if not (0 <= nb[0] < grid.width and 0 <= nb[1] < grid.height):
                continue
            cell = grid.get(*nb)
            if cell is None or cell.type == "goal":
                visited.add(nb)
                parent[nb] = pos
                q.append(nb)
    return None


def _path_to_actions(path: List[Tuple[int, int]], start_dir: int) -> List[int]:
    """Convert a sequence of (x,y) positions into MiniGrid actions."""
    actions = []
    d = start_dir
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        target = next(k for k, v in _DIR_VEC.items() if v == (dx, dy))
        turn = (target - d) % 4
        if turn == 1:
            actions.append(TURN_RIGHT)
        elif turn == 2:
            actions += [TURN_RIGHT, TURN_RIGHT]
        elif turn == 3:
            actions.append(TURN_LEFT)
        actions.append(FORWARD)
        d = target
    return actions


# ─────────────────────────────────────────────
#  Generate (obs, action) dataset
# ─────────────────────────────────────────────

def generate_demos(num_seeds: int = 50,
                   repeats: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each seed, find BFS-optimal actions and record (obs, action) pairs.
    *repeats* copies per seed give the dataset some bulk for stable training.
    """
    all_obs, all_act = [], []
    ok = 0

    for seed in range(num_seeds):
        env = make_env("MiniGrid-FourRooms-v0", seed=seed)
        env.reset(seed=seed)
        path = _bfs(env)
        if path is None or len(path) < 2:
            env.close()
            continue

        actions = _path_to_actions(path, int(env.unwrapped.agent_dir))
        # record (obs, action) by replaying
        for _ in range(repeats):
            obs, _ = env.reset(seed=seed)
            for a in actions:
                all_obs.append(obs.copy())
                all_act.append(a)
                obs, _, done, trunc, _ = env.step(a)
                if done or trunc:
                    break
        ok += 1
        env.close()

    print(f"Generated demos for {ok}/{num_seeds} seeds  "
          f"({len(all_obs):,} samples)")
    return np.array(all_obs, dtype=np.float32), np.array(all_act, dtype=np.int64)


# ─────────────────────────────────────────────
#  Quick evaluation of the pretrained model
# ─────────────────────────────────────────────

@torch.no_grad()
def _quick_eval(model: ActorCritic, num_seeds: int = 50,
                rollouts: int = 10, device: str = "cpu") -> Tuple[float, float]:
    model.eval()
    total_r, total_s, total_n = 0.0, 0, 0
    for seed in range(num_seeds):
        env = make_env("MiniGrid-FourRooms-v0", seed=seed)
        for _ in range(rollouts):
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            ep_r = 0.0
            while not (done or trunc):
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                dist, _, _ = model.get_dist_value(obs_t)
                obs, r, done, trunc, _ = env.step(dist.sample().item())
                ep_r += r
            total_r += ep_r
            total_s += int(ep_r > 0)
            total_n += 1
        env.close()
    model.train()
    return total_r / total_n, 100.0 * total_s / total_n


# ─────────────────────────────────────────────
#  Behavioural cloning training loop
# ─────────────────────────────────────────────

def pretrain(out_path: str = "pretrained.pt",
             epochs: int = 60,
             batch_size: int = 256,
             lr: float = 1e-3,
             ent_coef: float = 0.01,
             hidden_dim: int = 128,
             device: str = "cpu") -> ActorCritic:

    print("=" * 55)
    print("  PRETRAINING  (behavioural cloning on BFS demos)")
    print("=" * 55)

    obs_arr, act_arr = generate_demos()
    obs_dim = obs_arr.shape[1]
    act_dim = 7      # should be 7

    ds = TensorDataset(torch.tensor(obs_arr), torch.tensor(act_arr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = ActorCritic(obs_dim, act_dim, hidden_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for ob, ac in dl:
            ob, ac = ob.to(device), ac.to(device)
            logits = model.actor(ob)
            loss = ce(logits, ac)
            # entropy regulariser
            probs = torch.softmax(logits, dim=-1)
            ent   = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
            loss  = loss - ent_coef * ent

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1

        if ep % 5 == 0 or ep == 1:
            avg_r, succ = _quick_eval(model, device=device)
            print(f"  epoch {ep:3d}  loss {total_loss/n:.4f}  "
                  f"reward {avg_r:.3f}  success {succ:.1f}%")
            if succ >= 70.0:
                print(f"  ✓ reached target success rate")
                break

    # final eval
    avg_r, succ = _quick_eval(model, rollouts=50, device=device)
    print(f"\n  Final:  reward={avg_r:.3f}  success={succ:.1f}%")
    print(f"  Paper:  reward=0.469       success=70.4%\n")

    torch.save(model.state_dict(), out_path)
    print(f"  Saved → {out_path}")
    return model


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Pretrain on BFS expert demos")
    p.add_argument("--out",        default="pretrained.pt")
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int,   default=128)
    p.add_argument("--device",     default="cpu")
    args = p.parse_args()
    pretrain(args.out, args.epochs, args.batch_size, args.lr,
             hidden_dim=args.hidden_dim, device=args.device)


if __name__ == "__main__":
    main()
