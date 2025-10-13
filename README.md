# Coin Game — PPO Reinforcement Learning with Curriculum Training

This project implements a **Proximal Policy Optimization (PPO)** agent that learns to play a custom 2D platformer called **Coin Game**, where the goal is to **collect coins efficiently under time pressure**. The playable version of the game can be found in my Games and Toys repository.

The training process uses **curriculum learning** to automatically scale difficulty as the agent improves — teaching movement, navigation, and multi-platform jumping progressively.  
It also integrates **skill retention checks** to prevent forgetting earlier abilities as new ones are learned.

---

## Overview

### Objective
Train a reinforcement learning agent to:
- **Navigate**, **jump**, and **collect coins** within strict time limits  
- Progress through increasingly complex arena layouts  
- **Retain earlier skills** (like ground navigation) while learning new ones (like multi-level jumps)

The agent’s training follows a structured progression, starting from ground movement and culminating in full multi-platform mastery.

---

## Key Components

### `core.py` — Game Physics and Logic
Implements the deterministic 2D platformer from scratch:
- Physics engine with gravity, acceleration, and friction  
- Platform-based arenas (from flat ground to multi-tier layouts)  
- Collision detection and coin spawning logic  
- State tracking for agent position, velocity, coins, and timers

### `envCoinGame.py` — Gymnasium Environment
Wraps the game into a **Gymnasium-compatible environment** for reinforcement learning.  
Defines:
- **Action space:** 6 discrete actions (no-op, move, jump, or combinations)  
- **Observation space:** A normalized 35-dimensional vector of positions, velocities, timers, and platform layout  
- **Reward shaping:** A dense but carefully tuned set of rewards that balance exploration and goal completion

#### Reward Design Highlights
The reward system incentivizes both short-term and strategic behaviors:
- **Primary reward:** strong incentives for coin pickup and episode wins  
- **Speed bonus:** additional rewards for collecting quickly or winning with time to spare  
- **Exploration and jump learning:** bonuses for attempting jumps, reaching higher platforms, taking off from good positions, and landing successfully  
- **Navigation guidance:** encourages correct direction toward the coin and avoids “hovering near coin” local optima by rewarding **momentum and progression** rather than proximity alone

### `trainPPO.py` — PPO Training with Curriculum Learning
Implements a full PPO training pipeline using **Stable-Baselines3** and custom callbacks.

Key features:
- **Curriculum learning:** difficulty increases only when win-rate and episode thresholds are met  
- **Parallelized environments** for stable on-policy updates  
- **VecNormalize** for observation and reward normalization  
- **TensorBoard logging** for real-time metrics and stage transitions

#### Curriculum Progression (10 stages)
From **Ground Basic** movement to **Full Arena Mastery**, the curriculum raises:
- `coins_to_win` (more objectives per run)  
- `timer_budget` (less time per coin)  
- `arena_level` (more complex layouts)

### Skill Retention and Mixed Training
Two reinforcement mechanisms guard against forgetting:
- **Skill Retention Checks:** periodic short evaluations on earlier stages; if performance dips, the agent temporarily revisits those stages to refresh skills before resuming  
- **Mixed Training:** occasionally mixes previous-stage parameters during training for broader robustness

### `visualEval.py` — Visual Evaluation (pygame)
Minimal viewer to watch the trained agent:
- Toggle deterministic or stochastic actions  
- Restart, next, and pause controls  
- On-screen HUD for coins and timer; win/fail banners

---

## Training and Evaluation

### Train
```bash
python trainPPO.py
```
- Logs to `logs/`
- Saves checkpoints to `models/`

### Visualize
```bash
python visualEval.py
```
Ensure the trained artifacts exist:
```
models/ppo_coin_final.zip
models/vecnormalize.pkl
```

---

## Logging and Metrics

- **Win Rate Tracking** (including EWMA smoothing)  
- **Episode Stats** (rewards, lengths, success ratio)  
- **Curriculum Progress** (automatic stage transitions and thresholds)  
- **Final Summary** after training (totals, averages, recent performance)

View with:
```bash
tensorboard --logdir logs/
```

---
## PPO Agent Evaluation 🎮

Below is a short demo of the trained PPO agent playing the Coin Game. Apologies for the quality.

![PPO Coin Game Demo](./CoinGameRec.gif)

---
## Technical Deep Dive

### PPO Configuration
- Clipped surrogate objective, GAE advantages, and minibatch optimization  
- **VecNormalize** for observations and rewards to stabilize across stages

### Environment Normalization
Normalization stats are saved and loaded to keep **training versus evaluation** consistent.

### Curriculum Logic
Transitions are performance-triggered (win-rate and minimum episode count), not time-based.  
The callback adjusts:
- `coins_to_win`  
- `timer_budget`  
- `arena_level`  

### Retention Reinforcement
Periodic short tests on earlier stages:
- If retention falls below threshold → temporarily **revert** to that stage for a brief reinforcement period  
- Restore the current stage afterward  
- Prevents catastrophic forgetting while still advancing

### Reward System Dynamics
Multi-scale shaping that blends:
- **Immediate** (pickup, direction correctness)  
- **Exploration** (jump attempts, trying different approaches)  
- **Technique** (good jump positions, successful landings, vertical progress)  
- **Strategic** (speed to pickup/win, efficient transitions between coins)

---

## Future Work (WIP / TODO)

**Near-term improvements**
- Update `metricEval.py` (clean API, consistent metrics, side-by-side runs)  
- Add or fix support for saving and loading **“master” models** (canonical best checkpoint; easy export/use in eval)  
- Better tuning for **rewards** and **PPO hyperparameters** (sweeps and ablations with/without shaping)  
- Add **recording to GitHub** (artifacts or lightweight MP4/GIF renders from `visualEval`)

**Exploration and generalization**
- **Procedural arena generation** (parametric platform layouts; curriculum-ready)  
- **Evaluate on new arena layouts** (zero-shot and few-shot transfer; holdout test suites)

---

## Dependencies

```bash
pip install stable-baselines3[extra] gymnasium pygame numpy torch tensorboard
```
