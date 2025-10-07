"""
PPO Training Script for CoinGameEnv with Curriculum Learning.

This script trains a PPO agent on the CoinGameEnv environment using curriculum learning
to gradually increase difficulty as the agent improves. It uses 8 parallel environments
for efficient training and includes TensorBoard logging and model checkpointing.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn
from rl.envCoinGame import CoinGameEnv
import numpy as np
import os
import torch

# Global seed for full reproducibility
BASE_SEED = 42  # Changeable base seed for all randomness


def make_env(index: int, coins_to_win: int = 1, timer_budget: float = 60.0, curriculum_stage: int = 0):
    """
    Create a seeded environment factory for deterministic but unique rollouts.
    
    Args:
        index: Environment index (0 to n_envs-1)
        coins_to_win: Starting number of coins to win
        timer_budget: Starting timer budget per coin
        curriculum_stage: Current curriculum stage for arena modifications
        
    Returns:
        Environment factory function
    """
    def _init():
        # Create environment with curriculum starting parameters
        env = CoinGameEnv(max_steps=15000)  # Very long episodes to allow all curriculum stages
        env.set_params(coins_to_win=coins_to_win, timer_budget=timer_budget)
        
        # Set arena level based on curriculum stage
        env.set_params(arena_level=curriculum_stage)
        
        # Use deterministic but unique seed per environment
        # This ensures reproducible training with diverse rollouts
        env.reset(seed=BASE_SEED + index)
        return env
    return _init


def _apply_arena_modifications(env: CoinGameEnv, stage: int):
    """
    Apply arena modifications based on curriculum stage.
    
    Args:
        env: Environment to modify
        stage: Current curriculum stage (0-3)
    """
    if stage == 0:  # Super Easy - Larger platforms, more time
        # No modifications - use default arena
        pass
    elif stage == 1:  # Easy - Slightly smaller platforms
        # Could modify platform sizes here if we had that capability
        pass
    elif stage == 2:  # Medium - Standard arena
        # Standard arena
        pass
    elif stage == 3:  # Hard - Smaller platforms, more gaps
        # Could make platforms smaller, add gaps, etc.
        pass


class EpisodeStatsCallback(BaseCallback):
    """
    Callback to log episode statistics (rewards, lengths, win rates).
    """
    
    def __init__(self, verbose: int = 0):
        super(EpisodeStatsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        self.episode_failures = []
        self.episode_timeouts = []
        self.total_episodes = 0
        self.total_wins = 0
        self.total_failures = 0
        self.total_timeouts = 0
        
    def _on_step(self) -> bool:
        # Get episode info from training locals
        infos = self.locals.get("infos", [])
        
        
        for i, info in enumerate(infos):
            if isinstance(info, dict):
                # Check if episode ended by looking for win, fail_reason, or truncation
                episode_win = info.get('win', False)
                fail_reason = info.get('fail_reason', None)
                steps = info.get('steps', 0)
                
                # Episode ends when there's a win or fail reason (let env control truncation)
                episode_ended = False
                if episode_win:
                    episode_ended = True
                elif fail_reason is not None:
                    episode_ended = True
                
                if episode_ended:
                    # Track episode statistics
                    episode_length = steps
                    
                    # Update counters
                    self.total_episodes += 1
                    
                    if episode_win:
                        self.total_wins += 1
                        self.episode_wins.append(True)
                    else:
                        self.total_failures += 1
                        self.episode_wins.append(False)
                        
                        # Track failure reasons
                        if fail_reason == "timeout":
                            self.total_timeouts += 1
                            self.episode_timeouts.append(True)
                        else:
                            # Mark true timeouts separately; truncations are captured via info['truncated']
                            self.episode_timeouts.append(False)
                    
                    # Track other statistics
                    self.episode_lengths.append(episode_length)
                    self.episode_rewards.append(0.0)  # Placeholder - would need to track actual reward
                    
        
        # Log statistics every 1000 steps
        if self.n_calls % 1000 == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-50:])  # Last 50 episodes
            mean_length = np.mean(self.episode_lengths[-50:])
            win_rate = np.mean(self.episode_wins[-50:])
            
            self.logger.record("train/episode_reward", mean_reward)
            self.logger.record("train/ep_len_mean", mean_length)
            self.logger.record("train/win_rate", win_rate)
            
            if self.verbose > 0:
                print(f"Episode stats: reward={mean_reward:.2f}, length={mean_length:.1f}, win_rate={win_rate:.3f}")
        
        return True
    
    def print_final_statistics(self, curriculum_callback=None):
        """
        Print comprehensive training statistics at the end of training.
        """
        print(f"DEBUG: total_episodes = {self.total_episodes}")
        print(f"DEBUG: total_wins = {self.total_wins}")
        print(f"DEBUG: total_failures = {self.total_failures}")
        
        if self.total_episodes == 0:
            print("No episodes completed during training.")
            return
        
        # Calculate statistics
        win_rate = (self.total_wins / self.total_episodes) * 100
        timeout_rate = (self.total_timeouts / self.total_episodes) * 100
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        
        print("\n" + "="*60)
        print("🎯 FINAL TRAINING STATISTICS")
        print("="*60)
        print(f"📊 Total Episodes: {self.total_episodes}")
        print(f"🏆 Total Wins: {self.total_wins} ({win_rate:.1f}%)")
        print(f"❌ Total Failures: {self.total_failures} ({100-win_rate:.1f}%)")
        print(f"⏰ Timeout Failures: {self.total_timeouts} ({timeout_rate:.1f}%)")
        print(f"📏 Average Episode Length: {avg_length:.1f} steps")
        
        # Recent performance (last 100 episodes)
        if len(self.episode_wins) >= 100:
            recent_wins = sum(self.episode_wins[-100:])
            recent_win_rate = (recent_wins / 100) * 100
            print(f"🔥 Recent Performance (last 100 episodes): {recent_win_rate:.1f}% win rate")
        
        # Curriculum information
        if curriculum_callback is not None:
            current_stage = curriculum_callback.current_stage
            total_stages = len(curriculum_callback.curriculum_stages)
            stage_progress = (current_stage / (total_stages - 1)) * 100 if total_stages > 1 else 0
            
            # Get current stage parameters
            if current_stage < len(curriculum_callback.curriculum_stages):
                coins, timer, arena = curriculum_callback.curriculum_stages[current_stage]
                arena_names = ["Ground Only", "Ground+Middle", "Full Arena", "Jump Training Arena"]
                arena_name = arena_names[arena] if arena < len(arena_names) else f"Arena {arena}"
                print(f"🎓 Final Curriculum Stage: {current_stage + 1}/{total_stages} ({stage_progress:.1f}%)")
                print(f"   📍 Stage Parameters: {coins} coins, {timer}s timer, {arena_name}")
                
                if current_stage == total_stages - 1:
                    print("   🏆 MASTERED ALL LEVELS!")
                elif current_stage > 0:
                    print(f"   📈 Advanced {current_stage} stages from initial difficulty")
                else:
                    print("   🌱 Started at beginner level")
        
        print("="*60)


class CurriculumCallback(BaseCallback):
    """
    Curriculum learning callback that adjusts environment difficulty based on agent performance.
    
    This callback tracks the win rate across parallel environments and increases difficulty
    when the agent achieves a high enough success rate. The curriculum progresses through
    different combinations of coins_to_win and timer_budget parameters.
    """
    
    def __init__(self, check_freq: int = 5000, win_rate_threshold: float = 0.8, verbose: int = 1):
        """
        Initialize the curriculum callback.
        
        Args:
            check_freq: How often to check win rate (in steps)
            win_rate_threshold: Win rate threshold to trigger difficulty increase
            verbose: Verbosity level
        """
        super(CurriculumCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.win_rate_threshold = win_rate_threshold
        
        # Streamlined curriculum: (coins_to_win, timer_budget, arena_level)
        # 10 distinct stages emphasizing clear skill milestones
        self.curriculum_stages = [
            (1, 15.0, 0),   # A: Ground Basic (movement)
            (3, 15.0, 0),   # B: Ground Navigation (efficiency)
            (4, 20.0, 0),   # C: Ground Durability
            (1, 28.0, 3),   # D: Side Jump Intro (side arms; more time)
            (2, 24.0, 3),   # E: Side Jump Practice
            (1, 24.0, 1),   # F: Vertical Jump Intro (to middle)
            (4, 20.0, 1),   # G: Vertical Jump Multi (multi transitions)
            (3, 20.0, 2),   # H: Full Arena Intro
            (5, 15.0, 2),   # I: Full Arena Bridge (closer to eval)
            (7, 12.0, 2),   # J: Full Arena Mastery (final eval-like)
        ]
        self.current_stage = 0
        # Stage names aligned with curriculum_stages
        self.stage_names = [
            "Ground Basic",
            "Ground Navigation",
            "Ground Durability",
            "Side Jump Intro",
            "Side Jump Practice",
            "Vertical Jump Intro",
            "Vertical Jump Multi",
            "Full Arena Intro",
            "Full Arena Bridge",
            "Full Arena Mastery",
        ]
        
        # Track recent episodes for win rate calculation
        self.recent_episodes = []
        self.max_recent_episodes = 1000  # Keep last 1000 episodes for win rate calculation
        
        # Separate tracking for retention episodes (prevent win rate contamination)
        self.retention_episodes = []  # Track episodes during skill retention
        self.mixed_training_episodes = []  # Track episodes during mixed training
        self.is_retention_mode = False  # Flag to track if we're in retention mode
        self.is_mixed_training_mode = False  # Flag to track if we're in mixed training mode
        
        # Track jumping behavior for skill validation
        self.jump_attempts = []  # Track if agent attempted jumps in recent episodes
        self.jump_successes = []  # Track if jumps led to platform landings
        self.jump_actions_taken = []  # Track actual jump actions taken per episode (optional diagnostics)
        # Per-env episode jump counters; lazily initialized once training starts
        self.episode_jump_actions_per_env = None
        
        # Track evaluation performance
        self.eval_results = []  # Track evaluation win rates for each stage
        self.last_eval_check = 0  # Track when we last ran evaluation
        
        # Skill retention parameters (prevent catastrophic forgetting)
        self.skill_retention = True
        self.retention_check_freq = 5000  # Check skill retention every 5k steps (will be adjusted for higher stages)
        self.retention_threshold = 0.6  # Minimum win rate on previous stages
        self.retention_test_episodes = 10  # Episodes to test retention
        self.last_retention_check = 0  # Track when we last checked retention
        
        # Mixed training parameters (prevent forgetting by mixing stages)
        self.mixed_training = True
        self.mixed_training_freq = 10000  # Mix in previous stages every 10k steps
        self.mixed_episodes = 5  # Number of episodes to mix from previous stages
        self.last_mixed_training = 0  # Track when we last did mixed training
        
        # Curriculum smoothing parameters
        self.curriculum_smoothing = True
        self.smoothing_factor = 0.1  # How much to decay old episodes when advancing
        self.min_episodes_for_advance = 25  # Increased to ensure skill mastery
        
        # Exponential smoothing for win rate diagnostics
        self.ewma_win_rate = None  # type: ignore
        self.ewma_alpha = 0.1  # smoothing factor for EWMA (0<alpha<=1)
        self.short_tail_len = 200  # short window to monitor immediate performance
        
        # Cooldown after mode restores to avoid transient swings
        self.post_restore_cooldown_eps = 0
        
        # Skill-specific advancement requirements
        self.skill_requirements = {
            0: {"min_episodes": 20, "win_rate": 0.55, "eval_required": False},  # Ground Basic
            1: {"min_episodes": 25, "win_rate": 0.60, "eval_required": False},  # Ground Navigation
            2: {"min_episodes": 30, "win_rate": 0.60, "eval_required": False},  # Ground Durability
            3: {"min_episodes": 30, "win_rate": 0.55, "eval_required": False},  # Side Jump Intro
            4: {"min_episodes": 35, "win_rate": 0.60, "eval_required": False},  # Side Jump Practice
            5: {"min_episodes": 30, "win_rate": 0.60, "eval_required": False},  # Vertical Jump Intro
            6: {"min_episodes": 35, "win_rate": 0.65, "eval_required": True},   # Vertical Jump Multi
            7: {"min_episodes": 30, "win_rate": 0.60, "eval_required": False},  # Full Arena Intro
            8: {"min_episodes": 30, "win_rate": 0.62, "eval_required": False},  # Full Arena Bridge (5 coins, 15s)
            9: {"min_episodes": 25, "win_rate": 0.70, "eval_required": True},   # Full Arena Mastery (7 coins, 12s)
        }
        
    def _on_step(self) -> bool:
        """
        Called at each step during training.
        
        Returns:
            bool: True to continue training, False to stop
        """
        
        # Lazy init per-env jump counters
        if self.episode_jump_actions_per_env is None:
            try:
                num_envs = self.training_env.num_envs
            except Exception:
                num_envs = 1
            self.episode_jump_actions_per_env = [0 for _ in range(num_envs)]

        # Detect episodes every step (but only evaluate win rate every check_freq steps)
        self._detect_episodes()
        
        # Track jump actions for jump training stages
        # Note: Actual action tracking would require access to the action from the training loop
        # For now, we rely on episode length detection in _detect_episodes()
        
        # Handle skill reinforcement episodes
        if hasattr(self, 'reinforcement_episodes_remaining') and self.reinforcement_episodes_remaining > 0:
            self.reinforcement_episodes_remaining -= 1
            if self.reinforcement_episodes_remaining == 0:
                # Restore original params (no stage flip)
                if self.verbose > 0:
                    print(f"   🎯 Retention complete - restoring current stage parameters")
                self.is_retention_mode = False
                if hasattr(self, 'retention_restore_params'):
                    coins, timer, arena = self.retention_restore_params
                    self._update_all_environments(coins, timer, arena)
                    delattr(self, 'retention_restore_params')
                    # Re-enable VecNormalize training mode after retention burst
                    try:
                        if hasattr(self.training_env, 'training'):
                            self.training_env.training = True
                    except Exception:
                        pass
                delattr(self, 'reinforcement_episodes_remaining')
        
        # Handle mixed training episodes
        if hasattr(self, 'mixed_episodes_remaining') and self.mixed_episodes_remaining > 0:
            self.mixed_episodes_remaining -= 1
            if self.mixed_episodes_remaining == 0:
                # Restore original params (no stage flip)
                if self.verbose > 0:
                    print(f"   🎯 Mixed training complete - restoring current stage parameters")
                self.is_mixed_training_mode = False
                if hasattr(self, 'mixed_restore_params'):
                    coins, timer, arena = self.mixed_restore_params
                    self._update_all_environments(coins, timer, arena)
                    delattr(self, 'mixed_restore_params')
                    # Re-enable VecNormalize training mode after mixed burst
                    try:
                        if hasattr(self.training_env, 'training'):
                            self.training_env.training = True
                    except Exception:
                        pass
                delattr(self, 'mixed_episodes_remaining')
        
        # Check if it's time to evaluate curriculum
        if self.n_calls % self.check_freq == 0:
            eval_result = self._evaluate_curriculum()
            if eval_result is False:
                # Early stop requested by evaluation
                return False
            
        # Check skill retention periodically (prevent catastrophic forgetting)
        # Use longer intervals for higher stages to reduce interference
        effective_retention_freq = self.retention_check_freq
        if self.current_stage >= 8:  # Higher stages (8+)
            effective_retention_freq = self.retention_check_freq * 3  # 3x longer interval
        
        if (self.skill_retention and 
            self.n_calls - self.last_retention_check >= effective_retention_freq and
            self.current_stage > 0):  # Only check after stage 0
            self._check_skill_retention()
            self.last_retention_check = self.n_calls
            
        # Mixed training to prevent forgetting (mix in previous stages)
        # Use longer intervals for higher stages to reduce interference
        effective_mixed_freq = self.mixed_training_freq
        if self.current_stage >= 8:  # Higher stages (8+)
            effective_mixed_freq = self.mixed_training_freq * 2  # 2x longer interval
        
        if (self.mixed_training and 
            self.n_calls - self.last_mixed_training >= effective_mixed_freq and
            self.current_stage > 0):  # Only mix after stage 0
            self._do_mixed_training()
            self.last_mixed_training = self.n_calls
            
        return True
    
    def _check_skill_retention(self):
        """
        Check if the AI has retained skills from previous stages.
        If not, temporarily revert to easier stages for skill reinforcement.
        """
        if self.verbose > 0:
            print(f"\n🧠 SKILL RETENTION CHECK")
            print(f"   📊 Testing retention on previous stages...")
        
        # Test retention on a RANDOM SUBSET of previous stages to reduce overhead
        # Select up to 5 suitable previous stages uniformly at random (using numpy RNG)
        retention_scores = {}
        if self.current_stage <= 0:
            return
        candidate_stages = list(range(self.current_stage))  # all prior stages
        k = min(5, len(candidate_stages))
        sampled_stages = list(np.random.choice(candidate_stages, size=k, replace=False))
        # Sort for stable logging
        sampled_stages.sort()
        for stage in sampled_stages:
            coins, timer, arena = self.curriculum_stages[stage]
            win_rate = self._test_stage_performance(coins, timer, arena, self.retention_test_episodes)
            retention_scores[stage] = win_rate
            
            if self.verbose > 0:
                stage_names = getattr(self, 'stage_names', None)
                stage_label = stage_names[stage] if stage_names and stage < len(stage_names) else f"Stage {stage+1}"
                print(f"   📈 Stage {stage + 1} ({stage_label}): {win_rate:.2%} win rate")
        
        # Check if any previous stage has degraded performance
        degraded_stages = [stage for stage, score in retention_scores.items() 
                          if score < self.retention_threshold]
        
        if degraded_stages:
            if self.verbose > 0:
                print(f"   ⚠️  SKILL DEGRADATION DETECTED!")
                print(f"   📉 Stages with poor retention: {[s+1 for s in degraded_stages]}")
                print(f"   🔄 Temporarily reverting to skill reinforcement...")
            
            # Temporarily revert to the most degraded stage for skill reinforcement
            worst_stage = min(degraded_stages)
            self._reinforce_skills(worst_stage)
        else:
            if self.verbose > 0:
                print(f"   ✅ All previous skills retained!")
    
    def _test_stage_performance(self, coins, timer, arena, episodes):
        """
        Test performance on a specific stage configuration.
        """
        # This is a simplified version - in practice, you'd need to temporarily
        # change environment parameters and run episodes
        # For now, we'll use the current win rate as a proxy
        if len(self.recent_episodes) < episodes:
            return 0.0
        
        # Use recent episodes as a proxy for current performance
        recent_wins = sum(self.recent_episodes[-episodes:])
        return recent_wins / episodes
    
    def _reinforce_skills(self, target_stage):
        """
        Temporarily revert to a previous stage for skill reinforcement.
        """
        if self.verbose > 0:
            stage_names = getattr(self, 'stage_names', None)
            stage_label = stage_names[target_stage] if stage_names and target_stage < len(stage_names) else f"Stage {target_stage+1}"
            print(f"   🔄 Reinforcing skills on Stage {target_stage + 1} ({stage_label})")
        
        # Do NOT change current_stage; only swap environment params temporarily
        self.is_retention_mode = True
        # Save params to restore later
        restore_coins, restore_timer, restore_arena = self.curriculum_stages[self.current_stage]
        self.retention_restore_params = (restore_coins, restore_timer, restore_arena)
        # Switch envs to target stage params
        coins, timer, arena = self.curriculum_stages[target_stage]
        # Freeze normalization stats during retention burst
        try:
            if hasattr(self.training_env, 'training'):
                self.training_env.training = False
        except Exception:
            pass
        self._update_all_environments(coins, timer, arena)
        if self.verbose > 0:
            print(f"   📚 Skill reinforcement active for {self.retention_test_episodes * 2} episodes")
        # Run a short burst of reinforcement episodes
        self.reinforcement_episodes_remaining = self.retention_test_episodes * 2
    
    def _update_all_environments(self, coins, timer, arena):
        """
        Update all environments with new curriculum parameters.
        """
        try:
            # Access the vectorized environment
            vec_env = self.training_env
            
            # Handle VecNormalize wrapper
            if hasattr(vec_env, 'venv'):
                underlying_envs = vec_env.venv.envs
            else:
                underlying_envs = vec_env.envs
            
            # Update each environment
            for env in underlying_envs:
                if hasattr(env, 'set_params'):
                    env.set_params(coins_to_win=coins, timer_budget=timer, arena_level=arena)
                elif hasattr(env, 'env') and hasattr(env.env, 'set_params'):
                    env.env.set_params(coins_to_win=coins, timer_budget=timer, arena_level=arena)
            
            if self.verbose > 0:
                print(f"   🔄 Updated all environments: {coins} coins, {timer}s timer, arena {arena}")
                
        except Exception as e:
            if self.verbose > 0:
                print(f"   ⚠️  Warning: Could not update environments: {e}")
    
    def _do_mixed_training(self):
        """
        Perform mixed training by temporarily switching to previous stages.
        This helps prevent catastrophic forgetting by reinforcing old skills.
        """
        if self.verbose > 0:
            print(f"\n🔄 MIXED TRAINING")
            print(f"   📚 Reinforcing previous skills...")
        
        # Randomly select a previous stage to reinforce (include ALL previous stages)
        previous_stages = list(range(self.current_stage))
        if not previous_stages:
            return
            
        target_stage = int(np.random.choice(previous_stages))
        stage_names = getattr(self, 'stage_names', None)
        
        if self.verbose > 0:
            stage_label = stage_names[target_stage] if stage_names and target_stage < len(stage_names) else f"Stage {target_stage+1}"
            print(f"   🎯 Mixing in Stage {target_stage + 1} ({stage_label}) for {self.mixed_episodes} episodes")
        
        # Do NOT change current_stage; only swap env params temporarily
        self.is_mixed_training_mode = True
        # Save params to restore later
        restore_coins, restore_timer, restore_arena = self.curriculum_stages[self.current_stage]
        self.mixed_restore_params = (restore_coins, restore_timer, restore_arena)
        # Switch envs to target stage params
        coins, timer, arena = self.curriculum_stages[target_stage]
        # Freeze normalization stats during mixed burst
        try:
            if hasattr(self.training_env, 'training'):
                self.training_env.training = False
        except Exception:
            pass
        self._update_all_environments(coins, timer, arena)
        # Schedule return to original params after a short mix
        self.mixed_episodes_remaining = self.mixed_episodes
    
    def _detect_episodes(self):
        """
        Detect episodes every step and add them to recent_episodes.
        This runs every step to catch all episodes, not just during evaluation.
        """
        try:
            # Get recent episode results from training locals
            infos = self.locals.get("infos", [])
            
            # Extract per-step jump action and win status; only finalize on episode end
            for i, info in enumerate(infos):
                if isinstance(info, dict):
                    # Accumulate whether a jump action was taken this step for env i
                    if self.episode_jump_actions_per_env is not None:
                        if bool(info.get('jump_action', False)):
                            # Guard against index mismatch if num_envs changed
                            if i < len(self.episode_jump_actions_per_env):
                                self.episode_jump_actions_per_env[i] += 1
                    # Check if this is an episode-ending step (win or fail)
                    # Look for any episode-ending condition
                    episode_ended = False
                    episode_won = False
                    
                    # Check for win condition
                    if info.get('win', False):
                        episode_ended = True
                        episode_won = True
                    
                    # Check for fail condition
                    elif info.get('fail_reason') is not None:
                        episode_ended = True
                        episode_won = False
                    
                    # Check for timeout (time_left <= 0)
                    elif info.get('time_left', float('inf')) <= 0:
                        episode_ended = True
                        episode_won = False
                    
                    # Truncation: respect env's max_steps flag if provided
                    elif info.get('truncated', False):
                        episode_ended = True
                        episode_won = False
                    
                    # Track completed episodes (separate tracking for retention/mixed training)
                    if episode_ended:
                        if self.is_retention_mode:
                            # Track retention episodes separately
                            self.retention_episodes.append(episode_won)
                        elif self.is_mixed_training_mode:
                            # Track mixed training episodes separately
                            self.mixed_training_episodes.append(episode_won)
                        else:
                            # Track normal curriculum episodes
                            self.recent_episodes.append(episode_won)
                        
                        # Track jumping behavior for jump training stages using actual jump actions
                        if self.current_stage in [3, 4, 5]:  # Jump-oriented stages
                            attempted_jumps = False
                            if self.episode_jump_actions_per_env is not None and i < len(self.episode_jump_actions_per_env):
                                attempted_jumps = self.episode_jump_actions_per_env[i] > 0
                                # Record optional diagnostics
                                self.jump_actions_taken.append(self.episode_jump_actions_per_env[i])
                                # Reset per-env counter for next episode
                                self.episode_jump_actions_per_env[i] = 0
                            self.jump_attempts.append(attempted_jumps)
                            # Keep only recent window
                            if len(self.jump_attempts) > 100:
                                self.jump_attempts = self.jump_attempts[-100:]
                        
                        # Keep only recent episodes
                        if len(self.recent_episodes) > self.max_recent_episodes:
                            self.recent_episodes = self.recent_episodes[-self.max_recent_episodes:]
                            
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not detect episodes: {e}")
    
    def _evaluate_curriculum(self):
        """
        Evaluate current performance and potentially advance curriculum.
        """
        try:
            # Get current stage info
            stage_names = getattr(self, 'stage_names', [str(i) for i in range(len(self.curriculum_stages))])
            arena_names = ["Ground Only", "Ground+Middle", "Full Arena", "Jump Training Arena"]
            current_coins, current_timer, current_arena = self.curriculum_stages[self.current_stage]
            
            if self.verbose > 0:
                print(f"\n📊 CURRICULUM EVALUATION")
                print(f"   🎯 Current Stage: {self.current_stage + 1}/{len(self.curriculum_stages)} - {stage_names[self.current_stage]}")
                print(f"   📋 Parameters: {current_coins} coins, {current_timer}s timer, {arena_names[current_arena]}")
                print(f"   📈 Episodes tracked: {len(self.recent_episodes)}/{self.min_episodes_for_advance}")
            
            # Calculate win rate with smoothing
            if len(self.recent_episodes) >= 3:  # Need at least 3 episodes for win rate calculation
                win_rate = np.mean(self.recent_episodes)
                wins = sum(self.recent_episodes)
                total = len(self.recent_episodes)
                
                # Update EWMA-smoothed win rate
                if self.ewma_win_rate is None:
                    self.ewma_win_rate = float(win_rate)
                else:
                    self.ewma_win_rate = float(self.ewma_alpha * win_rate + (1 - self.ewma_alpha) * self.ewma_win_rate)
                # Short-tail win rate
                tail_n = min(self.short_tail_len, total)
                short_tail_win_rate = float(np.mean(self.recent_episodes[-tail_n:])) if tail_n > 0 else 0.0

                # Get skill-specific requirements for current stage
                stage_requirements = self.skill_requirements.get(self.current_stage, 
                                                              {"min_episodes": self.min_episodes_for_advance, 
                                                               "win_rate": self.win_rate_threshold})
                required_episodes = stage_requirements["min_episodes"]
                required_win_rate = stage_requirements["win_rate"]
                
                if self.verbose > 0:
                    print(f"   🏆 Win Rate: {win_rate:.2%} ({wins}/{total} episodes)")
                    print(f"   📉 EWMA Win Rate: {self.ewma_win_rate:.2%} (alpha={self.ewma_alpha})")
                    print(f"   🔎 Short-tail Win Rate ({tail_n} eps): {short_tail_win_rate:.2%}")
                    print(f"   🎯 Required: {required_win_rate:.2%} win rate, {required_episodes} episodes")
                
                # Log to TensorBoard as well
                try:
                    self.logger.record("train/win_rate_raw", float(win_rate))
                    self.logger.record("train/win_rate_ewma", float(self.ewma_win_rate))
                    self.logger.record("train/win_rate_short_tail", float(short_tail_win_rate))
                except Exception:
                    pass

                # Check if we should advance curriculum (with skill-specific requirements)
                can_advance = (win_rate >= required_win_rate and 
                                self.current_stage < len(self.curriculum_stages) - 1 and
                                len(self.recent_episodes) >= required_episodes)
                
                # Additional skill validation for jump training stages
                if can_advance and self.current_stage in [3, 4, 5]:  # Jump training stages
                    if self.verbose > 0:
                        print(f"   🔍 Validating jumping skills for stage {self.current_stage + 1}...")
                    if not self._validate_jumping_skills():
                        can_advance = False
                        if self.verbose > 0:
                            print(f"   🚫 Jump training incomplete - agent must demonstrate jumping skills")
                
                # Evaluation-based advancement for critical stages
                if can_advance and stage_requirements.get("eval_required", False):
                    if not self._validate_evaluation_performance():
                        can_advance = False
                        if self.verbose > 0:
                            print(f"   🚫 Evaluation performance insufficient - agent must pass evaluation tests")
                
                # Early-stop check: if at final stage, extremely high EWMA win rate and retention ok
                if self.current_stage == len(self.curriculum_stages) - 1:
                    try:
                        final_ok = (self.ewma_win_rate is not None and self.ewma_win_rate >= 0.95)
                        retention_ok = self._quick_full_retention_check()
                    except Exception:
                        final_ok = False
                        retention_ok = False
                    if final_ok and retention_ok:
                        if self.verbose > 0:
                            print("   🛑 Early stopping: final stage EWMA >= 95% and retention passed.")
                        # Signal training to stop
                        return False
                if can_advance:
                    self._advance_curriculum()
                elif self.current_stage >= len(self.curriculum_stages) - 1 and self.verbose > 0:
                    print(f"   🏆 Maximum curriculum stage reached! Agent mastered all levels.")
                elif self.verbose > 0:
                    # Check what's actually blocking advancement
                    if len(self.recent_episodes) < required_episodes:
                        print(f"   ⏳ Need more episodes: {len(self.recent_episodes)}/{required_episodes}")
                    elif win_rate < required_win_rate:
                        print(f"   ⏳ Need higher win rate: {win_rate:.2%}/{required_win_rate:.2%}")
                    else:
                        print(f"   ⏳ Win rate and episodes sufficient, but other requirements not met")
            else:
                if self.verbose > 0:
                    print(f"   ⏳ Not enough episodes yet: {len(self.recent_episodes)}/3")
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not evaluate curriculum: {e}")
                import traceback
                traceback.print_exc()
        # Continue training by default
        return True
    
    def get_current_win_rate(self):
        """
        Get the current win rate for debugging.
        
        Returns:
            float: Current win rate (0.0 to 1.0)
        """
        if len(self.recent_episodes) == 0:
            return 0.0
        return np.mean(self.recent_episodes)
    
    def _validate_jumping_skills(self):
        """
        Validate that the agent has learned jumping skills in jump training stages.
        
        Returns:
            bool: True if agent demonstrates jumping skills, False otherwise
        """
        if len(self.jump_attempts) < 10:  # Need sufficient data
            if self.verbose > 0:
                print(f"   🚫 Not enough jump data: {len(self.jump_attempts)} episodes")
            return False
        
        # Check if agent attempts jumps regularly (at least 30% of episodes)
        jump_attempt_rate = np.mean(self.jump_attempts[-20:])  # Last 20 episodes
        if self.verbose > 0:
            print(f"   📊 Jump attempt rate: {jump_attempt_rate:.2%} (need 30%)")
        
        if jump_attempt_rate < 0.3:  # Increased from 20% to 30%
            if self.verbose > 0:
                print(f"   🚫 Insufficient jump attempts: {jump_attempt_rate:.2%} < 30%")
            return False
        
        # Additional check: require longer episodes (suggesting actual jumping)
        recent_episodes = self.jump_attempts[-20:]
        long_episodes = sum(recent_episodes)
        if long_episodes < 6:  # Need at least 6 long episodes out of 20
            if self.verbose > 0:
                print(f"   🚫 Not enough long episodes: {long_episodes}/20 (need 6)")
            return False
        
        if self.verbose > 0:
            print(f"   ✅ Jump skills validated: {jump_attempt_rate:.2%} rate, {long_episodes} long episodes")
        return True
    
    def _validate_evaluation_performance(self):
        """
        Validate that the agent can pass evaluation tests for the current stage.
        
        Returns:
            bool: True if agent passes evaluation, False otherwise
        """
        # For now, use a simple heuristic: require high win rate for longer period
        if len(self.recent_episodes) < 50:  # Need more data for evaluation
            return False
        
        # Check if agent has maintained high performance for extended period
        recent_win_rate = np.mean(self.recent_episodes[-50:])  # Last 50 episodes
        if recent_win_rate < 0.80:  # Must maintain 80% win rate for evaluation
            return False
        
        # For jump training stages, also require jumping behavior
        if self.current_stage in [3, 4, 5]:
            if len(self.jump_attempts) < 20:
                return False
            jump_rate = np.mean(self.jump_attempts[-20:])
            if jump_rate < 0.3:  # Must attempt jumps in 30% of episodes
                return False
        
        return True

    def _quick_full_retention_check(self) -> bool:
        """
        Lightweight retention gate: sample up to 5 previous stages and require
        win rate >= retention_threshold on each using the existing proxy.
        """
        if self.current_stage <= 0:
            return True
        candidate_stages = list(range(self.current_stage))
        if not candidate_stages:
            return True
        k = min(5, len(candidate_stages))
        sampled = list(np.random.choice(candidate_stages, size=k, replace=False))
        for s in sampled:
            coins, timer, arena = self.curriculum_stages[s]
            wr = self._test_stage_performance(coins, timer, arena, self.retention_test_episodes)
            if wr < self.retention_threshold:
                return False
        return True
    
    def _advance_curriculum(self):
        """
        Advance to the next curriculum stage and update all environments.
        """
        self.current_stage += 1
        new_coins, new_timer, new_arena = self.curriculum_stages[self.current_stage]
        
        # Get stage names for clarity
        stage_names = getattr(self, 'stage_names', [str(i) for i in range(len(self.curriculum_stages))])
        arena_names = ["Ground Only", "Ground+Middle", "Full Arena", "Jump Training Arena"]
        
        if self.verbose > 0:
            print(f"\n🎓 CURRICULUM ADVANCEMENT!")
            print(f"   📈 Advanced to Stage {self.current_stage + 1}/{len(self.curriculum_stages)}: {stage_names[self.current_stage]}")
            print(f"   🎯 New Parameters: {new_coins} coins, {new_timer}s timer, {arena_names[new_arena]}")
            print(f"   🔄 Updating all {self.training_env.num_envs} environments...")
        
        # Adjust entropy coefficient by arena for exploration
        if new_arena == 0:  # Ground-only arenas
            # Standard entropy for ground navigation
            if hasattr(self.training_env, 'model') and hasattr(self.training_env.model, 'ent_coef'):
                self.training_env.model.ent_coef = 0.3  # Standard entropy for ground navigation
                if self.verbose > 0:
                    print(f"   🎲 Set entropy to 0.3 for ground navigation")
        elif new_arena == 1:  # Ground+Middle (vertical jump learning)
            if hasattr(self.training_env, 'model') and hasattr(self.training_env.model, 'ent_coef'):
                self.training_env.model.ent_coef = 0.5
                if self.verbose > 0:
                    print(f"   🎲 Increased entropy to 0.5 for vertical jump exploration")
        elif new_arena == 2:  # Full arena
            if hasattr(self.training_env, 'model') and hasattr(self.training_env.model, 'ent_coef'):
                self.training_env.model.ent_coef = 0.4
                if self.verbose > 0:
                    print(f"   🎲 Set entropy to 0.4 for full arena exploration")
        else:  # Jump training arena (side arms)
            if hasattr(self.training_env, 'model') and hasattr(self.training_env.model, 'ent_coef'):
                self.training_env.model.ent_coef = 0.6
                if self.verbose > 0:
                    print(f"   🎲 Increased entropy to 0.6 for side-jump exploration")
        
        # Update all parallel environments
        try:
            # VecNormalize doesn't pass through env_method calls, so access underlying envs directly
            if hasattr(self.training_env, 'venv'):  # VecNormalize wrapper
                underlying_envs = self.training_env.venv.envs
            else:  # Direct DummyVecEnv
                underlying_envs = self.training_env.envs
            
            # Update each environment individually
            for env in underlying_envs:
                env.set_params(
                    coins_to_win=new_coins, 
                    timer_budget=new_timer,
                    arena_level=new_arena
                )
            
            # Reset curriculum stats to prevent continuous advancement
            # The agent needs to prove itself at the new difficulty level
            self.recent_episodes = []
            self.jump_attempts = []
            self.jump_successes = []
            
            if self.verbose > 0:
                print(f"   ✅ Curriculum updated successfully!")
                print(f"   📊 Stats reset - agent must prove itself at new difficulty")
                print(f"   🎯 Current Stage: {stage_names[self.current_stage]} ({new_coins} coins, {new_timer}s, {arena_names[new_arena]})")
            
        except Exception as e:
            if self.verbose > 0:
                print(f"   ❌ Error updating curriculum: {e}")


def create_vectorized_env(num_envs: int = 8) -> VecNormalize:
    """
    Create a vectorized environment with multiple parallel CoinGameEnv instances.
    Each environment gets a deterministic but unique seed for reproducible training.
    Wraps with VecNormalize for observation and reward normalization.
    
    Args:
        num_envs: Number of parallel environments to create
        
    Returns:
        VecNormalize: Normalized vectorized environment
    """
    # Create list of seeded environment factory functions
    # Each environment gets a unique seed: BASE_SEED + index
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # Create vectorized environment
    vec_env = DummyVecEnv(env_fns)
    
    # Wrap with VecNormalize for observation and reward normalization
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,      # Normalize observations
        norm_reward=True,    # Normalize rewards
        clip_obs=10.0,      # Clip observations to [-10, 10]
        clip_reward=1000.0    # Clip rewards to [-1000, 1000] to allow collection rewards
    )
    
    return vec_env


def create_ppo_model(vec_env: VecNormalize, tensorboard_log: str) -> PPO:
    """
    Create and configure a PPO model with optimized hyperparameters.
    
    Args:
        vec_env: Vectorized environment for training
        tensorboard_log: Path for TensorBoard logging
        
    Returns:
        PPO: Configured PPO model
    """
    # Learning rate schedule for better learning progression
    lr_schedule = get_linear_fn(3e-4, 1e-4, 1.0)  # Start higher, end moderate
    # PPO hyperparameters optimized for the coin game
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        vec_env,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],  # Larger networks
            activation_fn=torch.nn.ReLU,  # ReLU activation
        ),
        n_steps=2048,          # Increased steps for more stable updates
        batch_size=512,        # Smaller batch size for more updates per step (2048/512 = 4 batches)
        n_epochs=4,            # Slightly increased epochs for better learning
        gamma=0.99,            # Slightly reduced discount factor
        gae_lambda=0.95,       # Reduced GAE lambda for more stable advantage estimation
        clip_range=0.2,        # Standard clipping for effective policy updates
        clip_range_vf=0.2,       # Standard value function clipping
        ent_coef=0.3,          # Higher entropy for more exploration, especially in jump training
        # Note: Entropy will be dynamically adjusted by CurriculumCallback for jump training stages
        vf_coef=0.7,           # Higher value function coefficient for better reward prediction
        max_grad_norm=0.5,     # Increased gradient norm for more stable learning
        learning_rate=lr_schedule,  # Use learning rate schedule for better progression
        verbose=1,             # Verbosity level
        tensorboard_log=tensorboard_log,  # TensorBoard logging
        device="auto",         # Automatically select CPU/GPU
        seed=BASE_SEED,        # Seed for reproducible training
    )
    
    return model


def setup_global_seeds():
    """
    Set global seeds for reproducibility.
    """
    # Set numpy seed for any numpy operations
    np.random.seed(BASE_SEED)
    
    print(f"   Global seeds set with BASE_SEED = {BASE_SEED}")


def list_master_models():
    """List available master models in masterModels directory."""
    if not os.path.exists("masterModels"):
        print("   📁 No masterModels directory found")
        return []
    
    master_files = [f for f in os.listdir("masterModels") if f.endswith('.zip')]
    if not master_files:
        print("   📁 No master models found")
        return []
    
    print("   📁 Available master models:")
    for i, model_file in enumerate(master_files, 1):
        model_name = model_file.replace('.zip', '')
        print(f"   {i}. {model_name}")
    
    return master_files

def setup_directories():
    """
    Create necessary directories for logging and model saving.
    """
    # Create logs directory for TensorBoard
    os.makedirs("logs", exist_ok=True)
    
    # Create models directory for checkpoints
    os.makedirs("models", exist_ok=True)
    
    # Create masterModels directory for master models
    os.makedirs("masterModels", exist_ok=True)
    
    print("✓ Created directories: logs/, models/, masterModels/")


def main():
    """
    Main training function that orchestrates the entire PPO training process.
    
    This function:
    1. Sets up directories and logging
    2. Creates vectorized environment
    3. Initializes PPO model with curriculum learning
    4. Trains the model for 1M timesteps
    5. Saves the final model
    """
    print("  Starting PPO Training for CoinGameEnv")
    print("=" * 50)
    
    # Setup global seeds for reproducibility
    setup_global_seeds()
    
    # Setup directories
    setup_directories()
    
    # Create vectorized environment with 8 parallel environments
    print("Creating vectorized environment...")
    vec_env = create_vectorized_env(num_envs=8)
    print("   Created 8 parallel environments")
    
    # Simple training mode selection
    print("\n" + "="*50)
    print("🎯 TRAINING MODE SELECTION")
    print("="*50)
    print("1. 🆕 Fresh Training - Start from scratch")
    print("2. 🔄 Continue from Master - Load a master model")
    print("="*50)
    
    choice = None
    while True:
        try:
            choice = input("Enter your choice (1/2): ").strip()
            if choice in ['1', '2']:
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n👋 Training cancelled.")
            return
    
    # Set initial curriculum parameters based on choice
    print("Setting initial curriculum parameters...")
    stage_names = ["Super Easy", "Easy", "Medium", "Jump Training Easy", "Jump Training", "Expert", "Hard", "Master", "Grandmaster", "Multi-Platform Bridge", "Multi-Platform", "Full Arena Bridge I", "Full Arena Bridge II", "Ultimate"]
    arena_names = ["Ground Only", "Ground+Middle", "Full Arena", "Jump Training Arena"]
    
    if choice == '2':  # Continue from Master
        print("🎯 Starting from advanced stage for master model...")
        initial_coins, initial_timer, initial_arena = 4, 20.0, 0  # Start at Stage 3: 4 coins, 20s, ground only
        print(f"   🎯 Advanced Start: 4/10 - {stage_names[3]}")
        print(f"   📋 Parameters: {initial_coins} coins, {initial_timer}s timer, {arena_names[initial_arena]}")
    else:
        initial_coins, initial_timer, initial_arena = 1, 15.0, 0  # Stage 0: 1 coin, 15s, ground only
        print(f"   🎯 Initial Stage: 1/10 - {stage_names[0]}")
        print(f"   📋 Parameters: {initial_coins} coins, {initial_timer}s timer, {arena_names[initial_arena]}")
    
    # Access underlying environments to set initial parameters
    if hasattr(vec_env, 'venv'):  # VecNormalize wrapper
        underlying_envs = vec_env.venv.envs
    else:  # Direct DummyVecEnv
        underlying_envs = vec_env.envs
    
    # Set initial parameters for all environments
    for i, env in enumerate(underlying_envs):
        env.set_params(
            coins_to_win=initial_coins,
            timer_budget=initial_timer,
            arena_level=initial_arena
        )
    print(f"   ✅ Initial curriculum parameters set for all {len(underlying_envs)} environments")
    
    if choice == '1':
        print("🆕 Starting fresh training...")
        model = create_ppo_model(vec_env, tensorboard_log="logs/ppo_coin")
        print("   PPO model created with optimized hyperparameters")
        
    elif choice == '2':
        print("🎓 Available master models:")
        master_files = list_master_models()
        if not master_files:
            print("❌ No master models found!")
            print("🆕 Falling back to fresh training...")
            model = create_ppo_model(vec_env, tensorboard_log="logs/ppo_coin")
            print("   PPO model created with optimized hyperparameters")
        else:
            print("Select master model by number:")
            while True:
                try:
                    master_choice = int(input(f"Enter master model number (1-{len(master_files)}): ")) - 1
                    if 0 <= master_choice < len(master_files):
                        selected_model = master_files[master_choice]
                        master_name = selected_model.replace('.zip', '')
                        specific_master_path = f"masterModels/{selected_model}"
                        break
                    else:
                        print(f"❌ Invalid choice. Please enter 1-{len(master_files)}")
                except (ValueError, KeyboardInterrupt):
                    print("❌ Invalid input or cancelled")
                    print("🆕 Falling back to fresh training...")
                    model = create_ppo_model(vec_env, tensorboard_log="logs/ppo_coin")
                    print("   PPO model created with optimized hyperparameters")
                    return
            
            if os.path.exists(specific_master_path):
                print(f"🔄 Loading master model: {master_name}")
                
                # Try to load master model's VecNormalize stats
                # Extract the descriptive part (e.g., "ground" from "ppo_coin_ground_master")
                model_parts = master_name.split('_')
                if len(model_parts) >= 3 and model_parts[-1] == 'master':
                    # Format: ppo_coin_ground_master -> ground
                    descriptive_part = model_parts[-2]
                else:
                    # Fallback: use the last part
                    descriptive_part = model_parts[-1]
                master_vecnormalize_path = f"masterModels/vecnormalize_{descriptive_part}.pkl"
                if os.path.exists(master_vecnormalize_path):
                    print(f"🔄 Loading master VecNormalize stats: {master_vecnormalize_path}")
                    # Load the master's VecNormalize stats
                    vec_env = VecNormalize.load(master_vecnormalize_path, vec_env.venv)
                    vec_env.training = True  # Set to training mode
                    print("✅ Loaded master VecNormalize stats successfully")
                else:
                    print(f"⚠️  Master VecNormalize stats not found: {master_vecnormalize_path}")
                    print("   ⚠️  WARNING: Using fresh normalization will cause catastrophic forgetting!")
                    print("   🎯 The AI will see observations on a completely different scale")
                    print("   💡 Consider creating a master model with proper VecNormalize stats")
                    print("   🔄 Continuing with fresh normalization (expect poor performance)")
                
                model = PPO.load(specific_master_path, env=vec_env)
                print(f"✅ Loaded master model from {specific_master_path}")
            else:
                print(f"❌ Master model {master_name} not found!")
                print("🆕 Falling back to fresh training...")
                model = create_ppo_model(vec_env, tensorboard_log="logs/ppo_coin")
                print("   PPO model created with optimized hyperparameters")
    
    # Create callbacks
    print("Setting up callbacks...")
    
    # Episode statistics callback
    episode_stats_callback = EpisodeStatsCallback(verbose=1)
    
    # Curriculum learning callback
    curriculum_callback = CurriculumCallback(
        check_freq=3000,       # Check win rate every 3k steps (more frequent for longer curriculum)
        win_rate_threshold=0.70,  # Slightly lower threshold for faster progression through 9 stages
        verbose=1              # Show curriculum progression
    )
    
    # Checkpoint callback for saving models
    # Scale save frequency by number of environments to get correct total steps
    n_envs = vec_env.num_envs
    save_freq = max(1, 200000 // n_envs)  # 200k total steps / 8 envs = 25k per env (more frequent saves)
    print(f"   Checkpoint frequency: {save_freq} steps per env = {save_freq * n_envs} total steps")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,   # Save every 200k total steps (25k per env)
        save_path="models/",   # Save to models directory
        name_prefix="ppo_coin"
    )
    
    # Combine callbacks
    callbacks = [episode_stats_callback, curriculum_callback, checkpoint_callback]
    print("   Callbacks configured: Episode stats + Curriculum learning + Model checkpointing")
    
    # Start training
    print("\n   Starting training for 12,000,000 timesteps...")
    print("   Monitor progress with: tensorboard --logdir logs/ppo_coin")
    print("   Models will be saved to models/ directory")
    print("-" * 50)
    
    try:
        # Train the model
        model.learn(
            total_timesteps=12_000_000,  # Extended for complex curriculum and stable learning
            callback=callbacks,
            progress_bar=True,  # Show progress bar
            tb_log_name="ppo_coin_training"
        )
        
        print("\n   Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n   Training interrupted by user")
        
    except Exception as e:
        print(f"\n   Training failed with error: {e}")
        raise
    
    finally:
        # Save final model
        print("Saving final model...")
        final_model_path = "models/ppo_coin_final.zip"
        model.save(final_model_path)
        print(f"   Final model saved to {final_model_path}")
        
        # Save VecNormalize stats
        print("Saving VecNormalize stats...")
        vec_normalize_path = "models/vecnormalize.pkl"
        vec_env.save(vec_normalize_path)
        print(f"   VecNormalize stats saved to {vec_normalize_path}")
        
        # Close environment
        vec_env.close()
        print("   Environment closed")
        
        # Print final training statistics
        print("   Printing final training statistics...")
        episode_stats_callback.print_final_statistics(curriculum_callback)
        
        print("\n   Training complete!")
        print("   View training logs: tensorboard --logdir logs/ppo_coin")
        print("   Load trained model: PPO.load('models/ppo_coin_final.zip')")


if __name__ == "__main__":
    main()
