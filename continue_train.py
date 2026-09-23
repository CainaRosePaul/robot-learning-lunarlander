from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def evaluate(model):
    env = Monitor(gym.make("LunarLander-v3"))
    env.reset(seed=123)
    try:
        return evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
    finally:
        env.close()


def main():
    output = Path("models") / "continued"
    output.mkdir(parents=True, exist_ok=True)
    env = Monitor(gym.make("LunarLander-v3"))
    try:
        model = PPO.load("models/ppo_lunarlander.zip", env=env, device="cpu")
        mean, std = evaluate(model)
        print(f"Baseline evaluation: {mean:.2f} +/- {std:.2f}", flush=True)
        model.learn(
            total_timesteps=300_000,
            reset_num_timesteps=False,
            callback=CheckpointCallback(
                save_freq=50_000, save_path=str(output), name_prefix="checkpoint"
            ),
        )
        model.save(output / "ppo_lunarlander")
        mean, std = evaluate(model)
        print(f"Continued evaluation: {mean:.2f} +/- {std:.2f}", flush=True)
        print(f"Saved model: {output / 'ppo_lunarlander.zip'}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
