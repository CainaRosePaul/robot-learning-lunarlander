"""Compare BC and PPO on 100 shared seeds outside the demonstration set."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from train_bc import BCPolicy


def evaluate(name, choose_action):
    env = gym.make("LunarLander-v3")
    episodes = []
    try:
        for seed in range(1000, 1100):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            steps = 0
            while True:
                obs, reward, terminated, truncated, _ = env.step(choose_action(obs))
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    # Use the installed environment's terminal rules, not a score threshold.
                    if terminated and reward == 100:
                        outcome = "landed"
                    elif terminated and env.unwrapped.game_over:
                        outcome = "crashed"
                    elif terminated:
                        outcome = "out_of_bounds"
                    else:
                        outcome = "timeout"
                    break
            episodes.append(dict(seed=seed, reward=total, steps=steps, outcome=outcome))
            if len(episodes) % 20 == 0:
                print(f"{name}: {len(episodes)}/100 episodes complete", flush=True)
    finally:
        env.close()
    rewards = [e["reward"] for e in episodes]
    result = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "outcomes": {k: sum(e["outcome"] == k for e in episodes)
                     for k in ("landed", "crashed", "out_of_bounds", "timeout")},
        "episodes": episodes,
    }
    print(name, {k: v for k, v in result.items() if k != "episodes"}, flush=True)
    return result


def main():
    checkpoint = torch.load("models/bc_policy.pt", map_location="cpu", weights_only=True)
    bc = BCPolicy(checkpoint["obs_dim"], checkpoint["n_actions"])
    bc.load_state_dict(checkpoint["model_state_dict"])
    bc.eval()

    def bc_action(obs):
        with torch.inference_mode():
            return int(bc(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    ppo = PPO.load("models/continued/ppo_lunarlander.zip", device="cpu")
    results = {"bc": evaluate("BC", bc_action),
               "ppo": evaluate("PPO", lambda obs: int(ppo.predict(obs, deterministic=True)[0]))}
    path = Path("data/bc_evaluation.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {path}", flush=True)


if __name__ == "__main__":
    main()
