from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


ENV_ID = "LunarLander-v3"

MODEL_PATH = Path("models/continued/ppo_lunarlander.zip")
DATA_PATH = Path("data/expert_demonstrations.npz")

N_EPISODES = 100
SEED = 123


def main():
    DATA_PATH.parent.mkdir(exist_ok=True)

    # Load our trained PPO expert
    model = PPO.load(MODEL_PATH)

    env = gym.make(ENV_ID)

    observations = []
    actions = []
    episode_rewards = []

    for episode in range(N_EPISODES):

        obs, _ = env.reset(seed=SEED + episode)

        done = False
        total_reward = 0.0

        while not done:

            # Ask PPO expert what it would do
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            # Save:
            # current state -> expert action
            observations.append(obs.copy())
            actions.append(int(action))

            # Perform action in environment
            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)

            done = terminated or truncated

        episode_rewards.append(total_reward)

        print(
            f"Episode {episode + 1}/{N_EPISODES} "
            f"reward = {total_reward:.2f}"
        )

    # Convert lists into NumPy arrays
    observations = np.asarray(
        observations,
        dtype=np.float32
    )

    actions = np.asarray(
        actions,
        dtype=np.int64
    )

    episode_rewards = np.asarray(
        episode_rewards,
        dtype=np.float32
    )

    # Save demonstrations
    np.savez_compressed(
        DATA_PATH,
        observations=observations,
        actions=actions,
        episode_rewards=episode_rewards
    )

    print()
    print(f"Collected {len(observations)} state-action pairs.")
    print(
        f"Expert demonstration reward: "
        f"{episode_rewards.mean():.2f} "
        f"+/- {episode_rewards.std():.2f}"
    )

    print(f"Saved dataset to: {DATA_PATH}")

    env.close()


if __name__ == "__main__":
    main()