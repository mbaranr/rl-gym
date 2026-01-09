import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

    def get_action(self, state):
        probs = self(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, states, actions):
        probs = self(states)
        dist = Categorical(probs)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()


def train_ppo(
    env,
    actor,
    critic,
    device,
    total_episodes=1000,
    max_steps=1000,
    gamma=0.99,
    lr=3e-4,
    eps_clip=0.2,
    k_epochs=4,
    update_freq=2048
):
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr)

    buffer = RolloutBuffer()
    episode_rewards = []
    timestep = 0

    for episode in range(1, total_episodes + 1):
        state = env.reset()
        ep_reward = 0

        for _ in range(max_steps):
            timestep += 1

            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            action, logprob = actor.get_action(state_tensor)
            value = critic(state_tensor)

            next_state, reward, done, _ = env.step(action.item())

            buffer.states.append(state_tensor)
            buffer.actions.append(action)
            buffer.logprobs.append(logprob)
            buffer.values.append(value)
            buffer.rewards.append(reward)
            buffer.dones.append(done)

            state = next_state
            ep_reward += reward

            if timestep % update_freq == 0:
                update_ppo(
                    actor, critic, buffer,
                    optimizer_actor, optimizer_critic,
                    gamma, eps_clip, k_epochs, device
                )
                buffer.clear()
                timestep = 0

            if done:
                break

        episode_rewards.append(ep_reward)
        print(f"Episode {episode} | Reward: {ep_reward:.1f}")

    return episode_rewards


def update_ppo(
    actor, critic, buffer,
    optimizer_actor, optimizer_critic,
    gamma, eps_clip, k_epochs, device
):
    states = torch.stack(buffer.states).to(device)
    actions = torch.stack(buffer.actions).to(device)
    old_logprobs = torch.stack(buffer.logprobs).to(device)
    rewards = buffer.rewards
    dones = buffer.dones
    values = torch.stack(buffer.values).to(device)

    # compute returns
    returns = []
    discounted = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted = 0
        discounted = reward + gamma * discounted
        returns.insert(0, discounted)

    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    advantages = returns - values.detach()

    for _ in range(k_epochs):
        logprobs, entropy = actor.evaluate(states, actions)
        state_values = critic(states)

        ratios = torch.exp(logprobs - old_logprobs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(state_values, returns)

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        loss.backward()
        optimizer_actor.step()
        optimizer_critic.step()


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)

    train_ppo(env, actor, critic, device)