import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return self.net(x)


def one_hot(state, state_dim):
    vec = np.zeros(state_dim, dtype=np.float32)
    vec[state] = 1.0
    return vec


def select_action(policy_net, state, epsilon, action_dim, device):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1, keepdim=True)
    else:
        return torch.tensor(
            [[random.randrange(action_dim)]],
            device=device,
            dtype=torch.long
        )


def optimize_model(
    policy_net,
    target_net,
    optimizer,
    memory,
    batch_size,
    gamma,
    device
):
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.cat(batch.state).to(device)
    actions = torch.cat(batch.action).to(device)
    rewards = torch.cat(batch.reward).to(device)
    next_states = torch.cat(batch.next_state).to(device)
    dones = torch.cat(batch.done).to(device)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q = rewards + gamma * next_q_values * (1 - dones)

    loss = F.smooth_l1_loss(q_values.squeeze(), target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_dqn(
    env,
    policy_net,
    target_net,
    device,
    episodes=2000,
    max_steps=100,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    memory_size=10000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    target_update=10
):
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(memory_size)

    epsilon = epsilon_start
    rewards_history = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        state = one_hot(state, env.observation_space.n)
        state = torch.tensor([state], device=device)

        total_reward = 0

        for _ in range(max_steps):
            action = select_action(
                policy_net, state, epsilon,
                env.action_space.n, device
            )

            next_state, reward, done, _, _ = env.step(action.item())
            next_state_oh = one_hot(next_state, env.observation_space.n)

            memory.push(
                state,
                action,
                torch.tensor([next_state_oh], device=device),
                torch.tensor([reward], device=device),
                torch.tensor([done], device=device, dtype=torch.float32)
            )

            state = torch.tensor([next_state_oh], device=device)
            total_reward += reward

            if len(memory) >= batch_size:
                optimize_model(
                    policy_net, target_net,
                    optimizer, memory,
                    batch_size, gamma, device
                )

            if done:
                break

        rewards_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(
                f"Episode {episode} | "
                f"Avg Reward (100): {avg_reward:.3f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    return rewards_history


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.n
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    train_dqn(env, policy_net, target_net, device)