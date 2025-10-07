"""
Gymnasium environment for the Coin game.

This environment wraps the game logic from game/core.py and provides a Gymnasium-compatible
interface for reinforcement learning training. It supports simultaneous actions through
MultiBinary action space and provides vectorized observations.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional

from game.core import Config, State, reset as core_reset, step as core_step, NOOP, LEFT, RIGHT, JUMP, _platform_tops


class CoinGameEnv(gym.Env):
    """
    Gymnasium environment for the Coin Timer game.
    
    This environment uses discrete actions for clear movement and jumping combinations.
    The agent must collect coins within a time limit to win the game.
    
    Action Space:
        Discrete(6): 0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=LEFT+JUMP, 5=RIGHT+JUMP
        
    Observation Space:
        Box(35,): [agent_center_x, agent_center_y, vx, vy, grounded, coin_center_x, coin_center_y, dx, dy, time_left_norm, coins_norm, platform_info...]
        - First 11 values: agent center state, coin center state, game state
        - Platform info: 6 platforms * 4 values (x, y, width, height) = 24 values
        All values normalized to [-1, 1] or [0, 1] range
        
    Rewards:
        +1 for coin pickup
        +5 for winning the game
        -1 for timeout or falling off the map
        
    Episode Termination:
        - Win: Collected required number of coins
        - Fail: Timeout or fell off the map
        - Truncation: Maximum steps reached
    """
    
    metadata = {"render_modes": ["none"]}
    
    def __init__(self, cfg: Optional[Config] = None, max_steps: int = 10_000):
        """
        Initialize the Coin Timer environment.
        
        Args:
            cfg: Game configuration. If None, uses default Config()
            max_steps: Maximum number of steps before truncation
        """
        super().__init__()
        
        # Store configuration and parameters
        self.cfg = cfg or Config()
        self.max_steps = max_steps
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(6)  # 0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=LEFT+JUMP, 5=RIGHT+JUMP
        # Observation space: [agent_x, agent_y, vx, vy, grounded, coin_x, coin_y, dx, dy, time_left_norm, coins_norm, platform_info...]
        # Platform info: 6 platforms * 4 values (x, y, width, height) = 24 values
        # Total: 11 + 24 = 35 values
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(35,), dtype=np.float32
        )
        
        # Initialize state variables
        self.state: Optional[State] = None
        self.rng: Optional[np.random.Generator] = None
        self._steps: int = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for deterministic behavior. If None, generates a random seed.
            options: Additional options (unused)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        # Call super().reset() to handle seed properly for vectorized environments
        if seed is None:
            # Generate a random seed when None to ensure vectorized trainers behave well
            seed = np.random.randint(0, 2**31 - 1)
        
        super().reset(seed=seed)
        
        # Reset step counter
        self._steps = 0
        
        # Reset game state using core logic
        self.state, self.rng = core_reset(self.cfg, seed)
        
        # Create initial observation
        obs = self._make_obs()
        
        # Create info dict with Python types for SB3 compatibility
        info = {
            "win": False,
            "fail_reason": None,
            "coins_collected": int(self.state.coins_collected),
            "time_left": float(self.state.time_left),
            "steps": int(self._steps),
            # Episode just reset: no jump yet
            "jump_action": False,
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Discrete action (0=NOOP, 1=LEFT, 2=RIGHT, 3=JUMP, 4=LEFT+JUMP, 5=RIGHT+JUMP)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert discrete action to core actions
        acts = []
        if action == 1 or action == 4:  # LEFT or LEFT+JUMP
            acts.append(LEFT)
        if action == 2 or action == 5:  # RIGHT or RIGHT+JUMP
            acts.append(RIGHT)
        if action == 3 or action == 4 or action == 5:  # JUMP or any +JUMP
            acts.append(JUMP)
        if not acts:  # no actions selected (action == 0)
            acts = [NOOP]

        # Track whether a jump action was requested this step (for curriculum metrics)
        did_jump = action in (3, 4, 5)
        
        # Execute step using core game logic
        self.state, events = core_step(self.cfg, self.state, acts, self.rng)
        
        # Increment step counter
        self._steps += 1
        
        # Calculate reward based on events
        reward = 0.0
        
        # Calculate distance to coin first (needed for rewards)
        agent_center_x = self.state.agent.x + self.cfg.agent_size_w / 2
        agent_center_y = self.state.agent.y + self.cfg.agent_size_h / 2
        coin_center_x = self.state.coin.x + 8  # 8 is half coin width
        coin_center_y = self.state.coin.y + 8  # 8 is half coin height
        distance_to_coin = np.sqrt((coin_center_x - agent_center_x)**2 + 
                                  (coin_center_y - agent_center_y)**2)
        
        # Event-based rewards (MASSIVE rewards for coin collection - most important!)
        if events.get("pickup"):
            reward += 100.0  # Strong reward for collecting a coin - this is the main goal!
            
            # Bonus for immediate collection when very close (encourage no jittering)
            if distance_to_coin < 20:  # Was very close when collected
                immediate_collection_bonus = 50.0  # Bonus for immediate collection
                reward += immediate_collection_bonus
            
            # Speed bonus for collecting quickly (encourage immediate collection)
            time_used = self.cfg.timer_budget - self.state.time_left
            time_remaining = self.state.time_left
            speed_bonus = (time_remaining / self.cfg.timer_budget) * 50.0  # Up to 50 bonus for fast collection
            reward += speed_bonus
            
            # Bonus for extremely fast collection (within first 20% of time)
            if time_remaining > self.cfg.timer_budget * 0.8:  # Collected in first 20% of time
                ultra_fast_bonus = 100.0  # Bonus for ultra-fast collection
                reward += ultra_fast_bonus
            
            # Progress bonus for collecting coins (encourage multi-coin navigation)
            progress_bonus = (self.state.coins_collected / self.cfg.coins_to_win) * 30.0  # Up to 30 bonus for progress
            reward += progress_bonus
            
            # Collection efficiency bonus (encourage immediate collection)
            collection_efficiency = 1.0 / max(1, self._steps)  # Reward for collecting quickly
            reward += collection_efficiency * 50.0  # Up to 50 bonus for quick collection
            
            # Bonus for collecting in very few steps
            if self._steps <= 10:  # Collected in 10 steps or less
                ultra_efficient_bonus = 100.0  # Bonus for ultra-efficient collection
                reward += ultra_efficient_bonus
            
            # Bonus for immediate collection when very close
            if distance_to_coin < 20:  # Was very close when collected
                immediate_collection_bonus = 20.0  # Bonus for immediate collection
                reward += immediate_collection_bonus
                
                # Bonus for super immediate collection
                if distance_to_coin < 10:  # Was extremely close when collected
                    super_immediate_bonus = 30.0  # Bonus for super immediate collection
                    reward += super_immediate_bonus
            
        if events.get("win"):
            reward += 50.0  # Strong reward for winning - this is the ultimate goal!
            
            
            
            # Speed bonus for winning quickly (encourage fast completion)
            time_used = self.cfg.timer_budget - self.state.time_left
            time_remaining = self.state.time_left
            win_speed_bonus = (time_remaining / self.cfg.timer_budget) * 50.0  # Up to 50 bonus for fast wins
            reward += win_speed_bonus
            
        if events.get("fail"):
            reward -= 1.0  # Restore timeout penalty
        
        # NO distance rewards - they encourage staying close without collecting
        # Only reward actual coin collection, not proximity
        
        # Progressive distance closing reward (encourage movement toward coin)
        # ONLY in ground-only arenas (arena_level 0) to avoid interfering with jump training
        if self.cfg.arena_level == 0 and hasattr(self, '_last_distance'):
            distance_improvement = self._last_distance - distance_to_coin
            if distance_improvement > 0:  # Getting closer
                closing_reward = distance_improvement * 2.0  # Strong reward for getting closer
                reward += closing_reward
            # REMOVED: Moving away penalty causes jittering
            # The AI needs to be able to move freely around coins
        self._last_distance = distance_to_coin  # Store for next step
        
        # NO proximity bonuses - they encourage staying close without collecting
        # Only reward actual coin collection, not proximity
        
        # REMOVED: Close penalties cause jittering in ground training
        # The AI needs to be able to get close to coins without being penalized
                
                
        
        # REMOVED: Time-based penalties cause jittering
        # The AI needs time to learn proper positioning and collection
        
        # REMOVED: Time waste penalty causes jittering
        # The AI needs time to position itself properly
        
        # Multi-coin navigation reward (encourage efficient navigation between coins)
        if self.cfg.coins_to_win > 1:
            # Reward for making progress towards the next coin
            coins_remaining = self.cfg.coins_to_win - self.state.coins_collected
            if coins_remaining > 0:
                navigation_bonus = (1.0 / coins_remaining) * 0.5  # Small bonus for efficient navigation
                reward += navigation_bonus
        
        
        # Bonus for winning with time remaining
        if events.get("win"):
            time_bonus = (self.state.time_left / self.cfg.timer_budget) * 1.0  # Up to 1.0 bonus for fast wins
            reward += time_bonus
        
        # Movement direction rewards (encourage correct navigation)
        # Apply only when not very close to the coin to avoid jitter near pickup
        if distance_to_coin > 20:
            if action == 1 or action == 4:  # Moving left (LEFT or LEFT+JUMP)
                if coin_center_x < agent_center_x:  # Coin is to the left
                    reward += 1.0  # Reward for correct direction
                elif distance_to_coin > 50:  # Only penalize when very far from coin
                    reward -= 0.1  # Small penalty for wrong direction
            if action == 2 or action == 5:  # Moving right (RIGHT or RIGHT+JUMP)
                if coin_center_x > agent_center_x:  # Coin is to the right
                    reward += 1.0  # Reward for correct direction
                elif distance_to_coin > 50:  # Only penalize when very far from coin
                    reward -= 0.1  # Small penalty for wrong direction
        
        # Continuous movement reward (encourage not stopping)
        if action == 1 or action == 2:  # Moving horizontally (LEFT or RIGHT)
            # Reward movement when far from coin
            if distance_to_coin > 30:  # When far from coin
                # Reward for any horizontal movement when coin is not directly above/below
                if abs(coin_center_y - agent_center_y) > 5:  # Coin is not at same level
                    reward += 0.5  # Reward for continuous movement
            # Do not reward movement when very close to coin to prevent jitter
            elif distance_to_coin > 20:  # When close but not too close
                reward += 0.2  # Small reward for movement when close
        
        # Efficient navigation reward (encourage moving toward coin when far away)
        if distance_to_coin > 50:  # Only when far from coin (avoid jittering when close)
            if action == 1 and coin_center_x < agent_center_x:  # Moving left toward coin
                reward += 1.0  # Navigation reward
            elif action == 2 and coin_center_x > agent_center_x:  # Moving right toward coin
                reward += 1.0  # Navigation reward
            elif action == 0:  # Stopping when far from coin
                reward -= 0.1  # Small penalty for not moving toward distant coin
        
        # Encourage jumping when coin is above and agent is grounded
        coin_is_above = coin_center_y < (agent_center_y - 10)  # add hysteresis so it's clearly above
        coin_is_directly_above = abs(coin_center_x - agent_center_x) < 20 and coin_center_y < (agent_center_y - 5)  # Directly above
        
        if self.state.agent.grounded and coin_is_above:
            if action in (3, 4, 5):
                reward += 0.5  # Clear incentive to initiate jump when needed
            elif action in (0, 1, 2) and self.cfg.arena_level > 0:  # Only penalize in jump training arenas
                reward -= 0.2  # Small penalty for not jumping when coin is clearly above
        
        # EXPLORATION REWARDS for jump training stages (encourage trying new behaviors)
        if self.cfg.arena_level > 0:  # Multi-platform arenas
            # Reward any jumping action (encourage exploration of jumping)
            if action in (3, 4, 5):  # Any jump action
                exploration_bonus = 0.3  # Keep strong reward for learning to jump
                reward += exploration_bonus
                
            # Reward moving away from coins (encourage exploration of different paths)
            if hasattr(self, '_last_distance') and distance_to_coin > self._last_distance:
                exploration_bonus = 0.2  # Keep reward for trying different approaches
                reward += exploration_bonus
                
            # Reward horizontal movement when coin is above (encourage getting running start)
            if coin_is_above and action in (1, 2):  # Moving horizontally when coin is above
                exploration_bonus = 0.4  # Keep strong reward for getting running start
                reward += exploration_bonus
                # Extra reward for building speed toward a jump (run-up)
                if abs(self.state.agent.vx) > 0.6 * self.cfg.vx_max and self.state.agent.grounded:
                    reward += 0.3
            
            # JUMP POSITION GUIDANCE - Teach optimal jump positions
            if coin_is_above and self.state.agent.grounded:
                # Calculate optimal jump distance (need running start)
                optimal_distance = 50  # Optimal distance for running start
                current_distance = abs(coin_center_x - agent_center_x)
                
                # Reward being at optimal distance for jumping
                if 30 <= current_distance <= 70:  # Sweet spot for jumping
                    position_bonus = 0.5  # Strong reward for being in good jump position
                    reward += position_bonus
                    
                    # Extra reward for jumping from optimal position
                    if action in (3, 4, 5):  # Jumping from good position
                        optimal_jump_bonus = 2.0  # Stronger reward for jumping from optimal position
                        reward += optimal_jump_bonus
                
                # Penalty for being too close to coin (can't get running start)
                elif current_distance < 30:
                    too_close_penalty = -0.3  # Penalty for being too close
                    reward += too_close_penalty
                    
                # Penalty for being too far (waste time getting to position)
                elif current_distance > 100:
                    too_far_penalty = -0.2  # Penalty for being too far
                    reward += too_far_penalty
            
            # RECOVERY GUIDANCE - Help AI recover from failed jumps
            if coin_is_above and not self.state.agent.grounded:
                # AI is airborne and coin is above - encourage moving toward coin
                if action == 1 and coin_center_x < agent_center_x:  # Moving left toward coin
                    recovery_bonus = 0.3  # Reward for moving toward coin while airborne
                    reward += recovery_bonus
                elif action == 2 and coin_center_x > agent_center_x:  # Moving right toward coin
                    recovery_bonus = 0.3  # Reward for moving toward coin while airborne
                    reward += recovery_bonus
            
            # GROUND RECOVERY - Help AI get back to ground when under platform
            if not coin_is_above and not self.state.agent.grounded:
                # AI is airborne but coin is not above - encourage getting back to ground
                if action == 0:  # NOOP to let gravity work
                    ground_recovery_bonus = 0.2  # Reward for letting gravity work
                    reward += ground_recovery_bonus
        
        # Special case: Coin is directly above - encourage moving away for running start
        if self.state.agent.grounded and coin_is_directly_above:
            # Reward moving away from directly above coin (for running start)
            if action == 1:  # Moving left when coin is directly above
                reward += 0.8  # Strong reward for getting running start
            elif action == 2:  # Moving right when coin is directly above
                reward += 0.8  # Strong reward for getting running start
            elif action == 0:  # Stopping when coin is directly above
                reward -= 0.5  # Penalty for not getting running start
            # Don't encourage jumping when directly below - need running start first
        
        # Encourage jumping after getting distance from directly above coin
        if self.state.agent.grounded and coin_is_above and not coin_is_directly_above:
            # Coin is above but not directly above - good position for jumping
            if action in (3, 4, 5):  # Jumping
                reward += 0.7  # Strong reward for jumping when in good position
            elif action == 0:  # Stopping when in good jump position
                reward -= 0.3  # Penalty for not jumping when in good position
        
        # Reward vertical progress toward an above coin
        current_abs_dy = abs((coin_center_y - agent_center_y) / self.cfg.H)
        last_abs_dy = getattr(self, "_last_abs_dy", None)
        if last_abs_dy is not None and coin_is_above:
            dy_improvement = last_abs_dy - current_abs_dy
            if dy_improvement > 0:
                reward += dy_improvement * 0.5  # Reward shrinking vertical gap when coin is above
        self._last_abs_dy = current_abs_dy

        # Detect takeoff and landing events to shape once per jump/landing
        was_grounded = getattr(self, "_last_grounded", True)
        last_grounded_y = getattr(self, "_last_grounded_y", None)
        # Takeoff bonus/penalty only on transition grounded->air
        if (not self.state.agent.grounded) and was_grounded:
            if action in (3, 4, 5):
                if coin_is_above:
                    reward += 0.5  # reward initiating a real jump
                else:
                    reward -= 0.3  # discourage pointless jumps
            # Track start of airtime
            self._airtime_steps = 0
        
        # Landing on a higher platform closer to coin
        if self.state.agent.grounded and not was_grounded:
            if last_grounded_y is not None and agent_center_y < last_grounded_y - 2:
                bonus = 0.4
                # Extra if also horizontally closer after the landing
                if getattr(self, "_last_horizontal_distance", None) is not None:
                    current_horizontal_distance = abs(coin_center_x - agent_center_x)
                    if current_horizontal_distance < self._last_horizontal_distance:
                        bonus += 0.1
                reward += bonus
                
                # Extra bonus for landing closer to an above coin
                if coin_is_above and not coin_is_directly_above:
                    landing_bonus = 0.5  # Bonus for successful jump toward above coin
                    reward += landing_bonus
                
                # MASSIVE reward for successful platform landing in jump training
                if self.cfg.arena_level > 0:  # Multi-platform arenas
                    platform_landing_bonus = 4.0  # Stronger reward for successful platform landing
                    reward += platform_landing_bonus
                    
                    # Extra bonus for landing on platform with coin above
                    if coin_is_above:
                        platform_coin_bonus = 1.0  # Extra bonus for landing on platform with coin above
                        reward += platform_coin_bonus
                    # Additional side-arm landing bonus specific to arena 3
                    if self.cfg.arena_level == 3:
                        # Side arms are at y=240; reward landings that are not ground level
                        if abs(agent_center_y - 240) < 10:
                            reward += 0.5
        # Track grounded state and last grounded Y
        self._last_grounded = bool(self.state.agent.grounded)
        if self.state.agent.grounded:
            self._last_grounded_y = agent_center_y
        self._last_horizontal_distance = abs(coin_center_x - agent_center_x)
        
        # REMOVED: Ultra-strong penalties cause jittering
        # The AI needs to be able to move freely near coins
        self._last_action = action

        # Strong reward for NOT jumping when at same level (encourage horizontal movement)
        if action == 0 or action == 1 or action == 2:  # NOOP, LEFT, or RIGHT (no jumping)
            height_diff = coin_center_y - agent_center_y
            if abs(height_diff) <= 2:  # Coin is at same level (with small tolerance)
                reward += 1.0  # Very strong reward for not jumping when at same level
                
                # Extra reward for efficient navigation when multiple coins remain
                if self.cfg.coins_to_win > 1:
                    efficiency_bonus = (1.0 / self.cfg.coins_to_win) * 0.3  # Small bonus for efficient navigation
                    reward += efficiency_bonus
        
        # Anti-jitter near coin: encourage commit to pickup without freezing just short
        if distance_to_coin < 20:
            last_act = getattr(self, "_last_action", None)
            horizontal_gap = abs(coin_center_x - agent_center_x)
            same_level = abs((self.state.coin.y + 8) - agent_center_y) <= 5
            # Penalize direction flip LEFT<->RIGHT when close
            if (last_act in (1, 2)) and (action in (1, 2)) and last_act != action:
                reward -= 0.2
            # If still a few pixels away horizontally, allow gentle approach even when close
            if horizontal_gap > 3:
                # Small reward for moving toward the coin within close range
                if action in (1, 2):
                    moving_left = (action == 1)
                    if (moving_left and coin_center_x < agent_center_x) or ((not moving_left) and coin_center_x > agent_center_x):
                        reward += 0.2
            else:
                # Within ~3 px horizontally; if aligned in height, reward brief NOOP to let collision occur
                if same_level:
                    if action == 0:
                        reward += 0.5
                    elif action in (1, 2):
                        reward -= 0.1
        
        # While airborne, discourage repeated JUMP actions without progress
        if not self.state.agent.grounded:
            if action in (3, 4, 5):
                reward -= 0.1  # pressing jump in air gives a small penalty
            # penalize lack of vertical progress toward an above coin
            if coin_is_above and last_abs_dy is not None:
                if dy_improvement <= 0:
                    reward -= 0.05
        else:
            # On ground: penalize unnecessary jumps when coin not clearly above
            # BUT only in ground-only arenas (arena_level 0) to avoid teaching anti-jumping behavior
            if action in (3, 4, 5) and not coin_is_above and self.cfg.arena_level == 0:
                height_diff = coin_center_y - agent_center_y
                if abs(height_diff) <= 2:
                    reward -= 2.0  # Strong penalty for jumping when coin is at same level
                else:
                    reward -= 1.0  # Strong penalty for jumping when coin is not above
            # In arenas with elevated platforms (arena_level > 0), encourage exploration through jumping
            elif action in (3, 4, 5) and self.cfg.arena_level > 0:
                # Check if coin is at same level and very close
                coin_center_y = self.state.coin.y + 8
                agent_center_y = self.state.agent.y + self.cfg.agent_size_h / 2
                height_diff = abs(coin_center_y - agent_center_y)
                
                if height_diff <= 5 and distance_to_coin < 30:  # Coin is at same level and close
                    reward -= 1.0  # Penalty for jumping when coin is at same level and close
                else:
                    # Small reward for attempting jumps in multi-platform arenas
                    reward += 0.1
                    
                    # Extra reward for jumping when coin is on a different level
                    if height_diff > 20:  # Coin is on different level
                        reward += 0.2  # Small incentive to jump when needed
        
        # Determine termination conditions
        terminated = bool(events.get("win")) or bool(events.get("fail"))
        
        # Force timeout if timer has run out (safety check)
        if self.state.time_left <= 0:
            terminated = True
            events["fail"] = "timeout"
        
        # Only truncate if episode hasn't already terminated
        truncated = not terminated and self._steps >= self.max_steps
        
        # Create observation
        obs = self._make_obs()
        
        # Create info dict with Python types for SB3 compatibility
        info = {
            "win": bool(events.get("win")),
            "fail_reason": events.get("fail"),
            "coins_collected": int(self.state.coins_collected),
            "time_left": float(self.state.time_left),
            "steps": int(self._steps),
            "truncated": bool(truncated),
            "max_steps": int(self.max_steps),
            # True if the policy issued a jump (or combo) this step
            "jump_action": bool(did_jump),
        }
        
        
        # Ensure Python types for SB3 compatibility
        reward = float(reward)
        terminated = bool(terminated)
        truncated = bool(truncated)
        
        
        return obs, reward, terminated, truncated, info
    
    def _make_obs(self) -> np.ndarray:
        """
        Create normalized observation vector from current state.
        
        Returns:
            Normalized observation array of shape (35,)
        """
        # Agent center position and velocity (normalized to [-1, 1])
        agent_center_x = self.state.agent.x + self.cfg.agent_size_w / 2
        agent_center_y = self.state.agent.y + self.cfg.agent_size_h / 2
        agent_x_norm = (agent_center_x / self.cfg.W) * 2 - 1  # [0, W] -> [-1, 1]
        agent_y_norm = (agent_center_y / self.cfg.H) * 2 - 1  # [0, H] -> [-1, 1]
        vx_norm = np.clip(self.state.agent.vx / self.cfg.vx_max, -1, 1)  # [-vx_max, vx_max] -> [-1, 1]
        
        # Vertical velocity scaling using config values
        vy_scale = self._get_vy_scale_factor()
        vy_norm = np.clip(self.state.agent.vy / vy_scale, -1, 1)
        
        # Grounded flag (0 or 1)
        grounded = float(self.state.agent.grounded)
        
        # Coin center position (normalized to [-1, 1])
        coin_center_x = self.state.coin.x + 8  # 8 is half coin width (assuming 16px)
        coin_center_y = self.state.coin.y + 8  # 8 is half coin height (assuming 16px)
        coin_x_norm = (coin_center_x / self.cfg.W) * 2 - 1
        coin_y_norm = (coin_center_y / self.cfg.H) * 2 - 1
        
        # Relative position (agent center to coin center) - clipped to [-1, 1]
        dx = np.clip((coin_center_x - agent_center_x) / self.cfg.W, -1, 1)
        dy = np.clip((coin_center_y - agent_center_y) / self.cfg.H, -1, 1)
        
        # Time left (normalized to [0, 1])
        time_left_norm = np.clip(self.state.time_left / self.cfg.timer_budget, 0, 1)
        
        # Coins collected (normalized to [0, 1])
        coins_norm = self.state.coins_collected / self.cfg.coins_to_win
        
        # Platform information (normalized to [-1, 1])
        platforms = _platform_tops(self.cfg)
        platform_info = []
        for platform in platforms:
            # Normalize platform coordinates: x, y, width, height
            platform_x_norm = (platform[0] / self.cfg.W) * 2 - 1  # [0, W] -> [-1, 1]
            platform_y_norm = (platform[1] / self.cfg.H) * 2 - 1  # [0, H] -> [-1, 1]
            platform_w_norm = (platform[2] / self.cfg.W) * 2  # [0, W] -> [0, 2] (width can be up to full width)
            platform_h_norm = (platform[3] / self.cfg.H) * 2  # [0, H] -> [0, 2] (height can be up to full height)
            
            platform_info.extend([platform_x_norm, platform_y_norm, platform_w_norm, platform_h_norm])
        
        # Combine into observation vector
        obs = np.array([
            agent_x_norm,
            agent_y_norm,
            vx_norm,
            vy_norm,
            grounded,
            coin_x_norm,
            coin_y_norm,
            dx,
            dy,
            time_left_norm,
            coins_norm
        ] + platform_info, dtype=np.float32)
        
        # Ensure correct shape and dtype for vectorized wrappers
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (35,):
            obs = obs.reshape(35)
        
        return obs
    
    def _get_vy_scale_factor(self) -> float:
        """
        Get the scale factor for vertical velocity normalization.
        
        Returns:
            Scale factor for normalizing vertical velocity to [-1, 1]
        """
        # Use jump_impulse magnitude as reference for upward velocity
        vy_max_up = abs(self.cfg.jump_impulse)
        
        # Estimate maximum downward velocity based on gravity and fall time
        # Assume agent can fall for about 1 second maximum
        vy_max_down = self.cfg.gravity * 1.0  # Maximum fall velocity
        
        # Use the larger of the two as the scale factor
        return max(vy_max_up, vy_max_down)
    
    def set_params(self, *, coins_to_win: Optional[int] = None, timer_budget: Optional[float] = None, arena_level: Optional[int] = None):
        """
        Set environment parameters for curriculum learning.
        
        Args:
            coins_to_win: Number of coins required to win (if provided)
            timer_budget: Time budget per coin in seconds (if provided)
            arena_level: Arena difficulty level (0=Easy, 1=Medium, 2=Hard, 3=Jump Training)
        """
        if coins_to_win is not None:
            self.cfg.coins_to_win = int(coins_to_win)
        if timer_budget is not None:
            self.cfg.timer_budget = float(timer_budget)
        if arena_level is not None:
            self.cfg.arena_level = int(arena_level)
    
    def render(self, mode: str = "none") -> None:
        """
        Render the environment (not implemented).
        
        Args:
            mode: Render mode (only "none" supported)
        """
        if mode != "none":
            raise NotImplementedError(f"Render mode '{mode}' not supported")
        # No rendering implemented - environment is headless
