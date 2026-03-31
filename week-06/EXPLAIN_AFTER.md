# Explanation and Analysis

## What We Implemented

We replicated three reinforcement learning algorithms on MiniGrid-FourRooms-v0
and compared them against Table 1 of the Polychromic PPO paper.

### 1. REINFORCE with Baseline

The simplest policy gradient method. We collect full episodes, compute
discounted returns G_t, subtract a learned value baseline V(s_t) to reduce
variance, and take one gradient step on the policy per update. A separate
critic network learns V(s) via MSE loss. This serves as our simplest baseline.

### 2. PPO (Proximal Policy Optimization)

The standard on-policy algorithm. We run 4 parallel environments for 128
steps each, compute GAE-λ advantages, then do 4 epochs of clipped minibatch
updates. The clipped surrogate objective prevents destructively large policy
changes. We also clip the value loss for stability.

### 3. Polychromic PPO

The novel contribution from the paper. It modifies PPO in two ways:

**Vine sampling**: After each rollout, we save the MiniGrid grid state at
p=2 equally-spaced points along env-0's trajectory. For each saved state,
we restore the environment and branch N=8 independent trajectories using the
current policy. This gives us multiple trajectories starting from the same state.

**Polychromic advantage**: From the N=8 vine trajectories, we form M=4 random
sets of n=4 trajectories each. Each set is scored using the polychromic
objective: f_poly = avg_reward × diversity, where diversity is the fraction
of trajectories that visit distinct sets of rooms. The bonus (best score minus
mean score) is added to the GAE advantages for the next W=5 timesteps. This
pushes the policy to learn diverse strategies, not just one high-reward path.

A KL penalty term anchors the policy to prevent instability from the
exploration pressure.

## Key Design Decisions

**No pretraining**: The paper pretrains on expert demonstrations, but we train
from scratch. This simplifies the pipeline and avoids needing a BFS expert or
the Minari dataset. The tradeoff is that our absolute numbers may differ from
the paper's, but the relative ordering (Poly-PPO > PPO ≈ REINFORCE) should hold.

**Additive bonus**: Rather than replacing GAE advantages at rollout states with
the polychromic advantage (as Algorithm 2 suggests), we add the poly bonus on
top of GAE. This is simpler to implement and ensures every timestep still
receives a meaningful gradient signal from the standard value function.

**State save/restore**: We checkpoint the full MiniGrid grid state (grid encoding,
agent position, direction) and restore it to branch vines. This is more reliable
than replaying actions from episode start, which can fail if the environment is
stochastic or episodes are very long.

## What to Expect

The paper reports for Four Rooms:
- REINFORCE: (0.639, 89.6%)
- PPO: (0.618, 89.2%)
- Poly-PPO: (0.666, 92.4%)

Since we train from scratch (no pretraining), absolute numbers will likely be
lower. The key result to look for is whether Poly-PPO achieves higher success
rate and reward than both baselines, demonstrating that the diversity-encouraging
polychromic objective helps the policy explore more effectively.

## Lessons Learned

1. **Diversity needs a signal**: Standard entropy bonuses add local randomness
   but don't encourage semantically different trajectories. The polychromic
   objective explicitly rewards visiting different rooms, which is a much
   stronger exploration signal for grid-world tasks.

2. **Vine sampling needs resets**: The algorithm requires resetting the
   environment to arbitrary states. This works in MiniGrid because we can
   directly manipulate the grid, but would not work in environments without
   reset capability (a limitation the paper acknowledges).

3. **KL penalty matters**: Without the KL term, the exploration pressure from
   the polychromic bonus can destabilise training. The penalty keeps the policy
   from drifting too far from its previous iteration.
