"""
Visual evaluation for the Coin Game PPO agent using pygame.
- Loads models/ppo_coin_final.zip
- Renders the current env.state each step
Controls:
  ESC / Q  -> quit
  R        -> restart current episode (new seed)
  N        -> next episode (new seed)
  P        -> pause / unpause
  D        -> toggle deterministic action (on/off)
"""

from __future__ import annotations
import time
import numpy as np
import pygame

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.envCoinGame import CoinGameEnv
# We import the private helper just to match platform drawing to core logic.
# (It's fine for tooling; if you later expose platforms in Config, switch to that.)
from game.core import _platform_tops  # type: ignore

# ---------- Drawing helpers ----------
COL_BG    = (18, 18, 22)
COL_PLAT  = (90, 90, 110)
COL_AGENT = (235, 235, 255)
COL_COIN  = (255, 215, 0)
COL_TEXT  = (200, 240, 255)
COL_WIN   = (130, 220, 150)
COL_FAIL  = (230, 120, 120)

def draw_frame(screen: pygame.Surface, env: CoinGameEnv, message: str = "", banner: str | None = None):
    screen.fill(COL_BG)
    cfg = env.cfg
    st  = env.state

    # Platforms (from core) - use the actual arena level from the environment
    for (x, y, w, h) in _platform_tops(cfg, cfg.arena_level):
        pygame.draw.rect(screen, COL_PLAT, pygame.Rect(x, y, w, h))

    # Agent
    pygame.draw.rect(
        screen, COL_AGENT,
        pygame.Rect(st.agent.x, st.agent.y, cfg.agent_size_w, cfg.agent_size_h)
    )

    # Coin (simple circle)
    pygame.draw.circle(screen, COL_COIN, (int(st.coin.x+8), int(st.coin.y+8)), 8)  # +8 to center 16x16

    # HUD
    font = pygame.font.SysFont(None, 22)
    hud1 = f"time_left: {st.time_left:4.1f}"
    hud2 = f"coins: {st.coins_collected}/{cfg.coins_to_win}"
    hud3 = message
    screen.blit(font.render(hud1, True, COL_TEXT), (10, 8))
    screen.blit(font.render(hud2, True, COL_TEXT), (10, 30))
    if hud3:
        screen.blit(font.render(hud3, True, COL_TEXT), (10, 52))

    # End banner
    if banner:
        big = pygame.font.SysFont(None, 36)
        color = COL_WIN if "WIN" in banner else COL_FAIL
        surf = big.render(banner, True, color)
        rect = surf.get_rect(center=(cfg.W // 2, 40))
        screen.blit(surf, rect)

# ---------- Main evaluation ----------
def main():
    # ----- Configure evaluation difficulty here if you want -----
    # These should match your target stage.
    target_coins = 10  # Hard evaluation (10 coins total)
    target_timer = 10.0  # 10 seconds per coin (100 seconds total possible)
    max_steps    = 15000  # Match training episode length
    model_path   = "models/ppo_coin_final.zip"

    # Create evaluation environment
    env = CoinGameEnv(max_steps=max_steps)
    env.set_params(coins_to_win=target_coins, timer_budget=target_timer, arena_level=2)  # 0=Easy, 1=Medium, 2=Hard, 3=Jump Training
    # Change arena_level to test different difficulties:
    # arena_level=0: Ground only (3 platform sections)
    # arena_level=1: Ground + middle platform  
    # arena_level=2: Full arena (ground + middle + upper platforms)
    # arena_level=3: Jump Training (continuous ground + side arms)
    
    # Arena level and curriculum parameters are set above; no debug prints
    
    # Create vectorized environment to match training setup
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    
    # Create vectorized environment (single env for evaluation)
    vec_env = DummyVecEnv([lambda: env])
    
    # Wrap with VecNormalize to match training environment
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,      # Normalize observations
        norm_reward=True,    # Normalize rewards
        clip_obs=10.0,      # Clip observations to [-10, 10]
        clip_reward=1000.0    # Clip rewards to [-1000, 1000] to allow collection rewards
    )
    
    # Load VecNormalize stats from training
    try:
        vec_env = VecNormalize.load("models/vecnormalize.pkl", DummyVecEnv([lambda: env]))
        vec_env.training = False  # Set to evaluation mode
        print("✅ Loaded VecNormalize stats successfully")
    except Exception as e:
        print(f"⚠️  Could not load VecNormalize stats: {e}")
        print("   Using fresh normalization (may cause inconsistent behavior)")
    
    # Load model with vectorized environment
    model = PPO.load(model_path, env=vec_env)
    
    # Model/environment spaces are available via attributes; no debug prints

    # Pygame
    pygame.init()
    cfg = env.cfg
    screen = pygame.display.set_mode((cfg.W, cfg.H))
    pygame.display.set_caption("Coin Game - Visual Evaluation")
    clock = pygame.time.Clock()

    deterministic = True
    paused = False
    episode_idx = 0

    def reset_episode(seed=None):
        nonlocal episode_idx
        if seed is None:
            seed = episode_idx
        # Reset the vectorized environment
        obs = vec_env.reset()
        episode_idx += 1
        return obs

    obs = reset_episode(seed=0)
    end_banner = None
    message = "R: restart | N: next ep | D: det toggle | P: pause | Esc/Q: quit"

    running = True
    while running:
        # Inputs
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_r:
                    obs = reset_episode()
                    end_banner = None
                elif e.key == pygame.K_n:
                    obs = reset_episode()
                    end_banner = None
                elif e.key == pygame.K_p:
                    paused = not paused
                elif e.key == pygame.K_d:
                    deterministic = not deterministic

        # Step policy if not paused / not ended
        if not paused and end_banner is None:
            if obs is None:
                obs = vec_env.reset()
            
            # Use vectorized environment (observations are automatically normalized)
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step the vectorized environment
            step_result = vec_env.step([action])
            
            # Handle different return formats (4 or 5 values)
            if len(step_result) == 4:
                obs, reward, terminated, info = step_result
                truncated = [False]  # No truncation in this version
            else:
                obs, reward, terminated, truncated, info = step_result

            if terminated[0] or truncated[0]:
                if info[0].get("win"):
                    end_banner = "WIN! (R to restart, N next)"
                else:
                    end_banner = f"FAIL ({info[0].get('fail_reason', 'timeout')}) - R/N"
                # Reset obs to None so it gets properly reset next time
                obs = None
        # Draw
        draw_frame(screen, env, message, end_banner)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()