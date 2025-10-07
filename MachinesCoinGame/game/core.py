"""
Pure Python 2D platform "coin timer" game core.
No pygame dependencies - designed to be wrapped by a Gymnasium environment.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any


# Action constants
NOOP = 0
LEFT = 1
RIGHT = 2
JUMP = 3


@dataclass
class Config:
    """Game configuration with all tunable parameters."""
    # Arena settings (pixels)
    W: int = 640
    H: int = 360
    gravity: float = 1500.0
    death_y: float = 410.0
    
    # Physics timestep
    dt: float = 1/60.0
    
    # Agent settings
    agent_size_w: int = 24
    agent_size_h: int = 24  # Make agent a square for simpler centering
    ax_ground: float = 3000.0  # horizontal acceleration on ground
    ax_air: float = 1500.0     # horizontal acceleration in air
    vx_max: float = 250.0      # maximum horizontal speed
    jump_impulse: float = -500.0  # upward jump velocity
    ground_friction: float = 0.85  # horizontal friction on ground
    
    # Game objectives
    timer_budget: float = 10.0  # seconds of time given per coin
    coins_to_win: int = 10      # number of coins needed to win
    
    # Arena configuration
    arena_level: int = 2         # 0=Easy, 1=Medium, 2=Hard, 3=Jump Training


@dataclass
class Agent:
    """Agent state with position, velocity, and physics flags."""
    x: float
    y: float
    vx: float
    vy: float
    grounded: bool = False


@dataclass
class Coin:
    """Coin state with position and spawn index."""
    x: float
    y: float
    spawn_index: int


@dataclass
class State:
    """Complete game state."""
    agent: Agent
    coin: Coin
    coins_collected: int
    time_left: float
    last_coin_spawn_index: int


def _platform_tops(cfg: Config, arena_level: int = 2) -> list[tuple[float, float, float, float]]:
    """
    Get platform rectangles as (x, y, width, height) based on arena level.
    
    Args:
        cfg: Game configuration
        arena_level: 0=Easy (ground only), 1=Medium (ground+middle), 2=Hard (full arena), 3=Jump Training (continuous ground + side arms)
    """
    if arena_level == 0:  # Easy: Just ground level (split into 3 parts)
        return [
            # Bottom platform split into 3 equal sections
            (0, 320, 213, 40),      # Left section
            (213, 320, 214, 40),    # Middle section  
            (427, 320, 213, 40),    # Right section
        ]
    elif arena_level == 1:  # Medium: Ground + middle platform
        return [
            # Bottom platform split into 3 equal sections
            (0, 320, 213, 40),      # Left section
            (213, 320, 214, 40),    # Middle section  
            (427, 320, 213, 40),    # Right section
            
            # Middle platform (center)
            (200, 240, 240, 15),
        ]
    elif arena_level == 2:  # Hard: Full arena (default)
        return [
            # Bottom platform split into 3 equal sections
            (0, 320, 213, 40),      # Left section
            (213, 320, 214, 40),    # Middle section  
            (427, 320, 213, 40),    # Right section
            
            # Middle platform (center)
            (200, 240, 240, 15),
            
            # Left upper platform (left arm)
            (0, 160, 200, 15),
            
            # Right upper platform (right arm)
            (440, 160, 200, 15),
        ]
    else:  # Level 3: Jump Training - Continuous ground + side arms
        return [
            # Continuous ground platform (no gaps)
            (0, 320, 640, 40),      # Full width ground platform
            
            # Left side arm platform (lowered to middle platform height)
            (0, 240, 200, 15),
            
            # Right side arm platform (lowered to middle platform height)
            (440, 240, 200, 15),
        ]


def _sample_coin_spawn(cfg: Config, rng: np.random.Generator, last_spawn: int) -> int:
    """
    Sample a coin spawn index, ensuring it's different from the last one.
    Returns index into platform list for spawn anchor.
    """
    platforms = _platform_tops(cfg, cfg.arena_level)
    available_indices = [i for i in range(len(platforms)) if i != last_spawn]
    return rng.choice(available_indices)


def _rects_intersect(ax: float, ay: float, aw: float, ah: float,
                    bx: float, by: float, bw: float, bh: float) -> bool:
    """
    Check if two axis-aligned rectangles intersect.
    Returns True if rectangles overlap, False otherwise.
    """
    return (ax < bx + bw and ax + aw > bx and 
            ay < by + bh and ay + ah > by)


def reset(cfg: Config, seed: int = 0) -> Tuple[State, np.random.Generator]:
    """
    Initialize a new game episode.
    
    Args:
        cfg: Game configuration
        seed: Random seed for deterministic behavior
        
    Returns:
        Tuple of (initial_state, random_generator)
    """
    # Create deterministic RNG
    rng = np.random.default_rng(seed)
    
    # Initialize agent at starting position (center of ground)
    agent = Agent(
        x=cfg.W // 2 - cfg.agent_size_w // 2,
        y=320 - cfg.agent_size_h,  # on top of ground platform (now 24px high)
        vx=0.0,
        vy=0.0,
        grounded=True
    )
    
    # Spawn first coin
    platforms = _platform_tops(cfg, cfg.arena_level)
    spawn_idx = rng.choice(len(platforms))
    platform = platforms[spawn_idx]
    
    # Position coin at platform center (aligned with agent center)
    coin = Coin(
        x=platform[0] + platform[2] // 2 - 8,  # 8 is half coin width (assuming 16px)
        y=platform[1] - 20,  # 20 pixels above platform (aligned with agent center)
        spawn_index=spawn_idx
    )
    
    # Initialize game state
    state = State(
        agent=agent,
        coin=coin,
        coins_collected=0,
        time_left=cfg.timer_budget,
        last_coin_spawn_index=spawn_idx
    )
    
    return state, rng


def step(cfg: Config, st: State, actions: list[int], rng: np.random.Generator) -> Tuple[State, Dict[str, Any]]:
    """
    Execute one physics timestep and return updated state and events.
    
    Args:
        cfg: Game configuration
        st: Current game state
        actions: List of actions to take (NOOP, LEFT, RIGHT, JUMP)
        rng: Random number generator
        
    Returns:
        Tuple of (new_state, events_dict)
    """
    # Create new state by copying current
    new_agent = Agent(
        x=st.agent.x,
        y=st.agent.y,
        vx=st.agent.vx,
        vy=st.agent.vy,
        grounded=st.agent.grounded
    )
    
    new_coin = Coin(
        x=st.coin.x,
        y=st.coin.y,
        spawn_index=st.coin.spawn_index
    )
    
    new_state = State(
        agent=new_agent,
        coin=new_coin,
        coins_collected=st.coins_collected,
        time_left=st.time_left - cfg.dt,
        last_coin_spawn_index=st.last_coin_spawn_index
    )
    
    events = {"pickup": False, "win": False, "fail": None}
    
    # === INPUT PROCESSING ===
    # Process all actions simultaneously
    horizontal_input = False
    
    for action in actions:
        if action == LEFT:
            accel = cfg.ax_ground if new_agent.grounded else cfg.ax_air
            new_agent.vx -= accel * cfg.dt
            horizontal_input = True
        elif action == RIGHT:
            accel = cfg.ax_ground if new_agent.grounded else cfg.ax_air
            new_agent.vx += accel * cfg.dt
            horizontal_input = True
    
    # Apply ground friction when not accelerating horizontally
    if new_agent.grounded and not horizontal_input:
        new_agent.vx *= cfg.ground_friction
    
    # Clamp horizontal velocity
    new_agent.vx = np.clip(new_agent.vx, -cfg.vx_max, cfg.vx_max)
    
    # === HORIZONTAL MOVEMENT & COLLISION ===
    # Move horizontally
    new_agent.x += new_agent.vx * cfg.dt
    
    # Screen boundary collision (prevent running off screen)
    if new_agent.x < 0:
        new_agent.x = 0
        new_agent.vx = 0
    elif new_agent.x > cfg.W - cfg.agent_size_w:
        new_agent.x = cfg.W - cfg.agent_size_w
        new_agent.vx = 0
    
    # Resolve horizontal collisions with platforms
    platforms = _platform_tops(cfg, cfg.arena_level)
    for platform in platforms:
        if _rects_intersect(new_agent.x, new_agent.y, cfg.agent_size_w, cfg.agent_size_h,
                           platform[0], platform[1], platform[2], platform[3]):
            # Collision detected - resolve based on approach direction
            if new_agent.vx > 0:  # Moving right, hit left side
                new_agent.x = platform[0] - cfg.agent_size_w
            else:  # Moving left, hit right side
                new_agent.x = platform[0] + platform[2]
            new_agent.vx = 0
    
    # === GRAVITY & JUMPING ===
    # Apply gravity
    new_agent.vy += cfg.gravity * cfg.dt
    
    # Handle jump input
    for action in actions:
        if action == JUMP and new_agent.grounded:
            new_agent.vy = cfg.jump_impulse
            break  # Only jump once per frame
    
    # Reset grounded flag - will be set if landing on top of platform
    new_agent.grounded = False
    
    # === VERTICAL MOVEMENT & COLLISION ===
    # Move vertically
    new_agent.y += new_agent.vy * cfg.dt
    
    # Resolve vertical collisions with platforms
    for platform in platforms:
        if _rects_intersect(new_agent.x, new_agent.y, cfg.agent_size_w, cfg.agent_size_h,
                           platform[0], platform[1], platform[2], platform[3]):
            # Collision detected - resolve based on approach direction
            if new_agent.vy > 0:  # Falling down, hit top of platform
                new_agent.y = platform[1] - cfg.agent_size_h
                new_agent.vy = 0
                new_agent.grounded = True
            else:  # Moving up, hit bottom of platform
                new_agent.y = platform[1] + platform[3]
                new_agent.vy = 0
    
    # === COIN PICKUP LOGIC ===
    # Check if agent collected the coin
    coin_size = 16  # Assume 16x16 coin
    
    # Check collision detection
    collision_detected = _rects_intersect(new_agent.x, new_agent.y, cfg.agent_size_w, cfg.agent_size_h,
                                        new_coin.x, new_coin.y, coin_size, coin_size)
    
    if collision_detected:
        events["pickup"] = True
        new_state.coins_collected += 1
        new_state.time_left = cfg.timer_budget  # Reset timer
        
        # Spawn new coin at different location
        platforms = _platform_tops(cfg, cfg.arena_level)
        new_spawn_idx = _sample_coin_spawn(cfg, rng, new_state.last_coin_spawn_index)
        new_platform = platforms[new_spawn_idx]
        
        new_coin.x = new_platform[0] + new_platform[2] // 2 - 8
        new_coin.y = new_platform[1] - 20
        new_coin.spawn_index = new_spawn_idx
        new_state.last_coin_spawn_index = new_spawn_idx
    
    # === TIMER & TERMINATION LOGIC ===
    # Check win condition
    if new_state.coins_collected >= cfg.coins_to_win:
        events["win"] = True
    
    # Check fail conditions
    if new_state.time_left <= 0:
        events["fail"] = "timeout"
    elif new_agent.y >= cfg.death_y:
        events["fail"] = "fall"
    
    return new_state, events
