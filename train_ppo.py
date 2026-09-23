from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

ENV_ID = "LunarLander-v3"
TOTAL_TIMESTEPS = 120_000
SEED = 42


def main():
    # 1. Create the training environment
    env = gym.make(ENV_ID)
    env.reset(seed=SEED)

    # 2. Create the PPO reinforcement-learning agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=SEED,
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
    )

    # 3. Train the agent
    print(f"Training PPO expert for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 4. Save the trained model
    model_path = MODEL_DIR / "ppo_lunarlander"
    model.save(model_path)

    print(f"Saved model to: {model_path}.zip")

    # 5. Create a separate environment for evaluation
    eval_env = gym.make(ENV_ID)

    # 6. Evaluate the trained PPO agent
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )

    # 7. Print the result
    print(
        f"Expert mean reward: "
        f"{mean_reward:.2f} +/- {std_reward:.2f}"
    )

    # 8. Clean up
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()