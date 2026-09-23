"""Watch PPO and BC on three matching LunarLander episode seeds."""

import gymnasium as gym
import pygame
import torch
from stable_baselines3 import PPO

from evaluate_agents import ENV_ID, PPO_PATH, load_bc


def main():
    ppo = PPO.load(PPO_PATH, device="cpu")
    bc = load_bc()

    def bc_action(obs):
        with torch.inference_mode():
            return int(bc(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    policies = [("PPO", lambda obs: int(ppo.predict(obs, deterministic=True)[0])),
                ("Behavioral Cloning", bc_action)]
    env = gym.make(ENV_ID, render_mode="human")
    try:
        for seed in range(1000, 1003):
            for name, choose_action in policies:
                obs, _ = env.reset(seed=seed)
                total = 0.0
                while True:
                    pygame.display.set_caption(
                        f"{name} | Seed {seed} | Reward {total:.1f} | Esc to exit"
                    )
                    if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                        return
                    obs, reward, terminated, truncated, _ = env.step(choose_action(obs))
                    total += float(reward)
                    if terminated or truncated:
                        print(f"{name}, seed {seed}: reward {total:.2f}", flush=True)
                        break
    except pygame.error:
        # Closing the simulation window ends the demo.
        return
    finally:
        env.close()


if __name__ == "__main__":
    main()
