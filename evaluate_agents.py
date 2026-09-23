from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO


ENV_ID = "LunarLander-v3"

PPO_PATH = Path("models/continued/ppo_lunarlander.zip")
BC_PATH = Path("models/bc_policy.pt")

N_EPISODES = 50
SEED = 1000


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=8, n_actions=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.network(x)


def evaluate_ppo():
    model = PPO.load(PPO_PATH)
    env = gym.make(ENV_ID)

    rewards = []

    for episode in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED + episode)

        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)

            done = terminated or truncated

        rewards.append(total_reward)

    env.close()

    return np.asarray(rewards)


def load_bc():
    checkpoint = torch.load(
        BC_PATH,
        map_location="cpu"
    )

    model = BCPolicy(
        obs_dim=checkpoint["obs_dim"],
        n_actions=checkpoint["n_actions"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def evaluate_bc():
    model = load_bc()
    env = gym.make(ENV_ID)

    rewards = []

    for episode in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED + episode)

        done = False
        total_reward = 0.0

        while not done:

            obs_tensor = torch.tensor(
                obs,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                logits = model(obs_tensor)

                action = int(
                    logits.argmax(dim=1).item()
                )

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)

            done = terminated or truncated

        rewards.append(total_reward)

    env.close()

    return np.asarray(rewards)


def main():

    print("Evaluating PPO...")
    ppo_rewards = evaluate_ppo()

    print("Evaluating Behavioral Cloning...")
    bc_rewards = evaluate_bc()

    print()
    print(
        f"PPO reward: "
        f"{ppo_rewards.mean():.2f} "
        f"+/- {ppo_rewards.std():.2f}"
    )

    print(
        f"BC reward: "
        f"{bc_rewards.mean():.2f} "
        f"+/- {bc_rewards.std():.2f}"
    )


if __name__ == "__main__":
    main()