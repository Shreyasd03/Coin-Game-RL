from __future__ import annotations
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.envCoinGame import CoinGameEnv

def run_one(model, env, seed=0, deterministic=True):
    obs, info = env.reset(seed=seed)
    total_r, steps = 0.0, 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, terminated, truncated, info = env.step(action)
        total_r += float(r); steps += 1
        if terminated or truncated:
            return total_r, steps, bool(info.get("win", False)), info

def main():
    # Create evaluation environment
    env = CoinGameEnv(max_steps=5000)
    env.set_params(coins_to_win=10, timer_budget=8.0)  # adjust if different
    
    # Create vectorized environment for VecNormalize
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize stats from training
    try:
        vec_env = VecNormalize.load("models/vecnormalize.pkl", vec_env)
        vec_env.training = False  # Set to evaluation mode
        vec_env.norm_reward = False  # Don't normalize rewards during evaluation
        print("Loaded VecNormalize stats from training")
    except FileNotFoundError:
        print("Warning: vecnormalize.pkl not found, using unnormalized environment")
    
    # Load your final model (or a checkpoint)
    model = PPO.load("models/ppo_coin_final.zip", env=vec_env)

    episodes = 10
    rets, lens, wins = [], [], 0
    for i in range(episodes):
        ret, steps, win, info = run_one(model, vec_env, seed=i, deterministic=True)
        rets.append(ret); lens.append(steps); wins += int(win)
        print(f"Ep {i:02d}: return={ret:.2f}, steps={steps}, win={win}")

    print("\nSummary")
    print(f"Avg return: {np.mean(rets):.2f} ± {np.std(rets):.2f}")
    print(f"Avg length: {np.mean(lens):.1f}")
    print(f"Win rate:   {wins}/{episodes}")
    vec_env.close()

if __name__ == "__main__":
    main()