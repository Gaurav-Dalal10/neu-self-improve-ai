import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# Policy Network πθ(a|s)
class Policy(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, act_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=-1)


# Compute discounted returns G_t
def compute_returns(rewards, gamma):

    returns = []
    R = 0

    for r in reversed(rewards):

        R = r + gamma * R
        returns.insert(0, R)

    return returns


# Sample action from policy
def select_action(policy, state):

    state = torch.tensor(state, dtype=torch.float32)

    probs = policy(state)

    dist = Categorical(probs)

    action = dist.sample()

    log_prob = dist.log_prob(action)

    return action.item(), log_prob


def main():

    # Create environment
    env = gym.make("CartPole-v1")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Initialize policy
    policy = Policy(obs_dim, act_dim)

    optimizer = optim.Adam(policy.parameters(), lr=2e-4)

    gamma = 0.99
    episodes = 1000

    for episode in range(episodes):

        state, _ = env.reset()

        log_probs = []
        rewards = []

        done = False

        # Run episode
        while not done:

            action, log_prob = select_action(policy, state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

            done = terminated or truncated

        # Compute return-to-go
        returns = compute_returns(rewards, gamma)

        # Compute REINFORCE loss
        loss = 0

        for log_prob, G in zip(log_probs, returns):

            loss += -log_prob * G

        # Update policy
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Print training progress
        print(f"Episode {episode} | Reward: {sum(rewards)}")

    env.close()


if __name__ == "__main__":
    main()