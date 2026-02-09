"""
Trading Agents for Multi-Agent Strategy
PPO-based agents specialized for different market regimes.
"""

import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TradingEnv(gym.Env):
    """
    Trading environment for PPO agents.
    
    Modes:
    - 'trend': Rewards following the trend (ADX-aware)
    - 'mean_rev': Rewards mean reversion trades (Bollinger-aware)
    - 'defensive': Rewards shorting during crashes (Crisis specialist)
    - 'recovery': Rewards aggressive long positions in recoveries
    """
    
    def __init__(self, df, config, mode='trend'):
        super(TradingEnv, self).__init__()
        self.df = df
        self.config = config
        self.mode = mode
        self.n_features = df.shape[1]
        self.window = config['WINDOW_SIZE']
        self.current_step = self.window
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window * self.n_features,),
            dtype=np.float32
        )
        
        self.data = df.values.astype(np.float32)
        self.cols = df.columns.tolist()
        self.idx_sma = self.cols.index('dist_sma200') if 'dist_sma200' in self.cols else -1
        self.idx_pct_b = self.cols.index('pct_b') if 'pct_b' in self.cols else -1
        self.idx_ret = self.cols.index('returns') if 'returns' in self.cols else 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        return self._get_obs(), {}
    
    def _get_obs(self):
        return self.data[self.current_step - self.window : self.current_step].flatten()
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0, True, False, {}
        
        act = np.clip(action[0], -1, 1)
        ret = self.data[self.current_step, self.idx_ret]
        
        reward = self._calculate_reward(act, ret)
        
        self.current_step += 1
        return self._get_obs(), reward, False, False, {}
    
    def _calculate_reward(self, action, ret):
        """Calculate mode-specific reward."""
        if self.mode == 'trend':
            reward = action * ret * 100
            # Bonus for aligning with SMA trend
            if self.idx_sma != -1:
                sma_dist = self.data[self.current_step, self.idx_sma]
                if sma_dist > 0 and action > 0.1:
                    reward += 0.05 * action
                elif sma_dist < 0 and action < -0.1:
                    reward += 0.05 * abs(action)
                    
        elif self.mode == 'mean_rev':
            if self.idx_pct_b != -1:
                pct_b = self.data[self.current_step, self.idx_pct_b]
                if pct_b < 0.3 and action > 0.5:
                    # Oversold, reward going long
                    reward = action * ret * 150
                elif pct_b > 0.7 and action < 0:
                    # Overbought, reward going short
                    reward = abs(action) * (-ret) * 100
                else:
                    reward = action * ret * 50
            else:
                reward = action * ret * 100
                
        elif self.mode == 'defensive':
            # Crisis specialist: aggressive shorting during crashes
            reward = action * ret * 100
            if ret < -0.01 and action < -0.1:
                # Double reward for correctly shorting a crash
                reward *= 2.0
            elif ret < -0.01 and action > 0.1:
                # Heavy penalty for catching falling knife
                reward -= action * 10.0
                
        elif self.mode == 'recovery':
            # Recovery specialist: aggressive long in recovery phases
            reward = action * ret * 100
            if action > 0.7:
                reward += 0.1 * action
            if ret > 0.01 and action > 0.5:
                reward *= 1.5
        else:
            reward = action * ret * 100
            
        return reward


class SeedEnsembleAgent:
    """
    Ensemble of PPO agents trained with different seeds.
    Predictions are averaged for robustness.
    """
    
    def __init__(self, env_fn, config, seeds, mode, verbose=False):
        self.models = []
        self.envs = []
        self.seeds = seeds
        self.mode = mode
        
        for seed in seeds:
            set_seed(seed)
            env = DummyVecEnv([env_fn])
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
            
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=config['LEARNING_RATE'],
                batch_size=config.get('BATCH_SIZE', 64),
                seed=seed
            )
            model.learn(total_timesteps=config['TRAINING_STEPS'])
            
            self.models.append(model)
            self.envs.append(env)
            
            if verbose:
                print(f"    Trained {mode} agent with seed {seed}")
    
    def predict(self, obs, deterministic=True):
        """
        Get ensemble prediction by averaging across seeds.
        
        Args:
            obs: Flattened observation array
            deterministic: Use deterministic policy
            
        Returns:
            Averaged action value
        """
        actions = []
        for model, env in zip(self.models, self.envs):
            obs_norm = env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=deterministic)
            actions.append(action[0])
        return np.mean(actions)
    
    def predict_all(self, obs, deterministic=True):
        """Get predictions from all seeds (for conflict detection)."""
        actions = []
        for model, env in zip(self.models, self.envs):
            obs_norm = env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=deterministic)
            actions.append(action[0])
        return np.array(actions)
    
    def get_env(self):
        """Get first environment (for normalization reference)."""
        return self.envs[0]
