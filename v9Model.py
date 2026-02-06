"""
V9 Trading Model - Enhanced with Seed Control & 4-Agent Ensemble

Key Improvements over V8:
1. Deterministic seeding for reproducibility
2. Seed ensemble averaging (3 models per agent)
3. 4-agent architecture: Trend, MeanRev, Defensive, Recovery
4. Weighted ensemble voting instead of hard regime switching
"""

import yfinance as yf
import pandas as pd
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import itertools
import warnings
import os

warnings.filterwarnings('ignore')

# --- SEED CONTROL ---
def set_seed(seed: int):
    """Fix all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- V9 CONFIGURATION ---
DEFAULT_CONFIG = {
    # Assets
    "TICKERS": ["SPY", "IWM", "^VIX", "SHY"], 
    "TARGET_ASSET": "SPY",
    
    # Dates
    "TRAIN_START": "2015-01-01",
    "TRAIN_END": "2023-12-31",
    "TEST_START": "2024-01-02",
    "TEST_END": "2025-05-05",
    
    # Crisis Training Periods (for Defensive agent)
    "CRISIS_PERIODS": [
        ("2018-10-01", "2019-01-01"),  # Volmageddon
        ("2020-01-01", "2020-05-01"),  # COVID
        ("2022-01-01", "2022-12-31"),  # Bear market
        ("2018-01-01", "2018-04-01"),  # Feb 2018 correction
    ],
    
    # Recovery Training Periods (for Recovery agent)
    "RECOVERY_PERIODS": [
        ("2019-01-01", "2019-06-01"),  # Post-Volmageddon
        ("2020-04-01", "2020-09-01"),  # Post-COVID rally
        ("2023-01-01", "2023-07-01"),  # 2023 recovery
    ],
    
    # Seed Ensemble
    "SEEDS": [42, 123, 456],
    "USE_SEED_ENSEMBLE": True,
    
    # Hyperparameters
    "WINDOW_SIZE": 60, 
    "ADX_THRESHOLD": 30,
    "JUMP_THRESHOLD": 3.5, 
    "TARGET_VOL": 0.40,
    "MOMENTUM_LOOKBACK": 50,
    "MOMENTUM_THRESHOLD": 0.15,
    
    # Weighted Ensemble
    "USE_WEIGHTED_ENSEMBLE": True,
    "BASE_WEIGHTS": {
        'trend': 0.35,
        'mean_rev': 0.25,
        'defensive': 0.25,
        'recovery': 0.15
    },
    
    # Fixed Params
    "BB_STD": 2.0,
    "LEARNING_RATE": 3e-4,
    "BATCH_SIZE": 64,
    "TRAINING_STEPS": 50000, 
    "INITIAL_BALANCE": 100000,
    "FEES": 0.0005,
    "BORROW_RATE": 0.0002, 
    "ACTION_SCALER": 5.0,
    "MAX_LEVERAGE": 1.0, 
    "USE_VOL_TARGETING": True,
    "SMA_TREND_FILTER": True,
    "LONG_ONLY": False,
    
    # Bear trend position limits
    "BEAR_TREND_MIN_POS": 0.25,
    "BEAR_TREND_MAX_POS": 0.75,
    
    # Output settings
    "SAVE_PLOTS": True,
    "PLOTS_DIR": "./plots"
}

# --- DATA PROCESSOR ---
class DataProcessor:
    def __init__(self, tickers, config):
        self.tickers = list(set(tickers))
        self.config = config
        
    def download(self, start, end):
        try:
            if self.config['TARGET_ASSET'] not in self.tickers:
                self.tickers.append(self.config['TARGET_ASSET'])
                
            print(f"Fetching data for: {self.tickers} ({start} to {end})")
            data = yf.download(self.tickers, start=start, end=end, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]: data = data.xs('Close', level=0, axis=1)
                elif 'Adj Close' in data.columns.levels[0]: data = data.xs('Adj Close', level=0, axis=1)
                elif 'Close' in data.columns.levels[1]: data = data.xs('Close', level=1, axis=1)
            
            if isinstance(data, pd.Series): 
                data = data.to_frame()
                if self.config['TARGET_ASSET'] not in data.columns:
                    data.columns = [self.config['TARGET_ASSET']]
            
            ohlc = yf.download(self.config['TARGET_ASSET'], start=start, end=end, progress=False)
            return data, ohlc
        except Exception as e:
            print(f"Data Download Error: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def calculate_lee_mykland(self, df, window=20):
        log_ret = np.log(df / df.shift(1))
        abs_ret = np.abs(log_ret)
        bv_terms = abs_ret * abs_ret.shift(1)
        local_vol = np.sqrt((np.pi / 2) * bv_terms.rolling(window=window).mean())
        l_stat = log_ret / (local_vol + 1e-9)
        return l_stat.fillna(0)

    def add_features(self, df, ohlc):
        target = self.config['TARGET_ASSET']
        if df.empty or target not in df.columns: return pd.DataFrame()
        
        df = df.copy()
        df['returns'] = df[target].pct_change()
        
        # Jump Detection
        l_stat = self.calculate_lee_mykland(df[target], window=20)
        df['l_stat'] = l_stat
        
        # Trend indicators (ADX)
        high = ohlc['High']
        low = ohlc['Low']
        close = ohlc['Close']
        df['tr'] = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        df['dm_plus'] = np.where((high - high.shift(1)) > (low.shift(1) - low), np.maximum(high - high.shift(1), 0), 0)
        df['dm_minus'] = np.where((low.shift(1) - low) > (high - high.shift(1)), np.maximum(low.shift(1) - low, 0), 0)
        
        window = 14
        df['tr_s'] = df['tr'].rolling(window).mean()
        df['dp_s'] = df['dm_plus'].rolling(window).mean()
        df['dm_s'] = df['dm_minus'].rolling(window).mean()
        df['di_plus'] = 100 * (df['dp_s'] / df['tr_s'])
        df['di_minus'] = 100 * (df['dm_s'] / df['tr_s'])
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].rolling(window).mean()
        
        # Mean Reversion (Bollinger Bands)
        sma = df[target].rolling(20).mean()
        std = df[target].rolling(20).std()
        df['bb_width'] = (std * 2 * 2) / sma
        df['pct_b'] = (df[target] - (sma - 2*std)) / (4*std)
        
        # Crisis indicators
        if '^VIX' in df.columns: df['vix_norm'] = (df['^VIX'] - 15) / 40
        else: df['vix_norm'] = 0
            
        sma200 = df[target].rolling(200).mean()
        df['dist_sma200'] = (df[target] - sma200) / sma200
        
        # Volatility
        df['realized_vol_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['vol_percentile'] = df['realized_vol_20d'].rolling(252).rank(pct=True)
        
        # Momentum
        momentum_lookback = self.config.get('MOMENTUM_LOOKBACK', 50)
        df['momentum_50d'] = df[target].pct_change(momentum_lookback)
        
        # Recovery indicator (drawdown from high)
        rolling_high = df[target].rolling(50).max()
        df['drawdown'] = (df[target] - rolling_high) / rolling_high
        df['recovery_signal'] = (df['drawdown'] > -0.05) & (df['drawdown'].shift(10) < -0.10)
        
        return df.fillna(0)

    def get_data(self, start, end):
        df, ohlc = self.download(start, end)
        return self.add_features(df, ohlc)

    def get_crisis_data(self):
        dfs = []
        for s, e in self.config['CRISIS_PERIODS']:
            d, o = self.download(s, e)
            if not d.empty:
                dfs.append(self.add_features(d, o))
        if dfs:
            return pd.concat(dfs).reset_index(drop=True).fillna(0)
        return pd.DataFrame()
    
    def get_recovery_data(self):
        dfs = []
        for s, e in self.config['RECOVERY_PERIODS']:
            d, o = self.download(s, e)
            if not d.empty:
                dfs.append(self.add_features(d, o))
        if dfs:
            return pd.concat(dfs).reset_index(drop=True).fillna(0)
        return pd.DataFrame()

# --- TRADING ENVIRONMENT ---
class TradingEnv(gym.Env):
    def __init__(self, df, config, mode='trend'):
        super(TradingEnv, self).__init__()
        self.df = df
        self.config = config
        self.mode = mode 
        self.n_features = df.shape[1]
        self.window = config['WINDOW_SIZE']
        self.current_step = self.window
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window * self.n_features,), dtype=np.float32)
        
        self.data = df.values.astype(np.float32)
        self.cols = df.columns.tolist()
        self.idx_sma = self.cols.index('dist_sma200') if 'dist_sma200' in self.cols else -1
        self.idx_pct_b = self.cols.index('pct_b') if 'pct_b' in self.cols else -1
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        return self._get_obs(), {}
    
    def _get_obs(self):
        return self.data[self.current_step-self.window : self.current_step].flatten()

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0, True, False, {}
            
        act = np.clip(action[0], -1, 1)
        ret = self.data[self.current_step, 0] 
        
        reward = 0
        if self.mode == 'trend':
            # Reward directional bets that align with market
            reward = act * ret * 100
            if self.idx_sma != -1:
                sma_dist = self.data[self.current_step, self.idx_sma]
                if sma_dist > 0 and act > 0.1: reward += 0.05 * act 
                elif sma_dist < 0 and act < -0.1: reward += 0.05 * abs(act)
                
        elif self.mode == 'mean_rev':
            # Reward mean reversion: buy low pct_b, sell high pct_b
            if self.idx_pct_b != -1:
                pct_b = self.data[self.current_step, self.idx_pct_b]
                if pct_b < 0.3 and act > 0.5:  # Oversold, go long
                    reward = act * ret * 150
                elif pct_b > 0.7 and act < 0:  # Overbought, reduce
                    reward = abs(act) * (-ret) * 100
                else:
                    reward = act * ret * 50
            else:
                reward = act * ret * 100

        elif self.mode == 'defensive':
            # Reward protective behavior during negative returns
            reward = act * ret * 100
            if ret < -0.01 and act < 0.1:  # Down day, low exposure = good
                reward += 0.5 * (1 - act)
            elif ret < -0.02 and act > 0.5:  # Big down day with high exposure = bad
                reward -= act * 20
            
        elif self.mode == 'recovery':
            # Reward aggressive buying during recovery phases
            reward = act * ret * 100
            if act > 0.7:  # High conviction long
                reward += 0.1 * act
            if ret > 0.01 and act > 0.5:  # Up day with exposure = bonus
                reward *= 1.5
            
        self.current_step += 1
        return self._get_obs(), reward, False, False, {}

# --- SEED ENSEMBLE AGENT ---
class SeedEnsembleAgent:
    """Wrapper that trains multiple models with different seeds and averages predictions"""
    
    def __init__(self, env_fn, config, seeds, mode):
        self.models = []
        self.envs = []
        self.seeds = seeds
        
        for seed in seeds:
            set_seed(seed)
            env = DummyVecEnv([env_fn])
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
            
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=0, 
                learning_rate=config['LEARNING_RATE'],
                seed=seed
            )
            model.learn(total_timesteps=config['TRAINING_STEPS'])
            
            self.models.append(model)
            self.envs.append(env)
    
    def predict(self, obs, deterministic=True):
        """Get ensemble averaged prediction"""
        actions = []
        for model, env in zip(self.models, self.envs):
            obs_norm = env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=deterministic)
            actions.append(action[0])
        
        # Average across seeds
        return np.mean(actions)
    
    def get_env(self):
        """Return first env for normalization reference"""
        return self.envs[0]

# --- V9 ENSEMBLE MANAGER ---
class EnsembleManager:
    def __init__(self, config):
        self.config = config
        self.dp = DataProcessor(config['TICKERS'], config)
        self.agents = {}
        
        if config.get('SAVE_PLOTS', False):
            os.makedirs(config.get('PLOTS_DIR', './plots'), exist_ok=True)
        
    def train_specialists(self, verbose=True):
        if verbose: print("\n=== 1. TRAINING V9 SPECIALIST AGENTS (4-Agent Ensemble) ===")
        
        # Get training data
        trend_data = self.dp.get_data(self.config['TRAIN_START'], self.config['TRAIN_END'])
        crisis_data = self.dp.get_crisis_data()
        recovery_data = self.dp.get_recovery_data()
        
        seeds = self.config['SEEDS'] if self.config['USE_SEED_ENSEMBLE'] else [42]
        
        # 1. Trend Agent
        if verbose: print(f"  Training TREND agent with seeds {seeds}...")
        self.agents['trend'] = SeedEnsembleAgent(
            env_fn=lambda: TradingEnv(trend_data, self.config, mode='trend'),
            config=self.config,
            seeds=seeds,
            mode='trend'
        )
        
        # 2. Mean Reversion Agent
        if verbose: print(f"  Training MEAN_REV agent with seeds {seeds}...")
        self.agents['mean_rev'] = SeedEnsembleAgent(
            env_fn=lambda: TradingEnv(trend_data, self.config, mode='mean_rev'),
            config=self.config,
            seeds=seeds,
            mode='mean_rev'
        )
        
        # 3. Defensive Agent (trained on crisis data)
        if not crisis_data.empty:
            if verbose: print(f"  Training DEFENSIVE agent with seeds {seeds}...")
            self.agents['defensive'] = SeedEnsembleAgent(
                env_fn=lambda: TradingEnv(crisis_data, self.config, mode='defensive'),
                config=self.config,
                seeds=seeds,
                mode='defensive'
            )
        else:
            if verbose: print("  WARNING: No crisis data, using trend agent as defensive fallback")
            self.agents['defensive'] = self.agents['trend']
        
        # 4. Recovery Agent (trained on recovery periods)
        if not recovery_data.empty:
            if verbose: print(f"  Training RECOVERY agent with seeds {seeds}...")
            self.agents['recovery'] = SeedEnsembleAgent(
                env_fn=lambda: TradingEnv(recovery_data, self.config, mode='recovery'),
                config=self.config,
                seeds=seeds,
                mode='recovery'
            )
        else:
            if verbose: print("  WARNING: No recovery data, using trend agent as recovery fallback")
            self.agents['recovery'] = self.agents['trend']
        
        if verbose: print("=== TRAINING COMPLETE ===\n")

    def calculate_weights(self, indicators):
        """Calculate dynamic weights based on market indicators"""
        adx = indicators.get('adx', 20)
        vix = indicators.get('vix', 15)
        momentum = indicators.get('momentum', 0)
        sma_dist = indicators.get('sma_dist', 0)
        drawdown = indicators.get('drawdown', 0)
        
        base = self.config['BASE_WEIGHTS'].copy()
        
        # Strong trend -> increase trend weight
        if adx > 30:
            base['trend'] += 0.15
            base['mean_rev'] -= 0.10
            base['defensive'] -= 0.05
        
        # High VIX -> increase defensive weight
        if vix > 25:
            base['defensive'] += 0.20
            base['trend'] -= 0.10
            base['recovery'] -= 0.10
        
        # Strong positive momentum -> increase trend/recovery
        if momentum > 0.15:
            base['trend'] += 0.10
            base['recovery'] += 0.10
            base['defensive'] -= 0.15
            base['mean_rev'] -= 0.05
        
        # In recovery phase (rebounding from drawdown)
        if drawdown > -0.05 and sma_dist < 0:
            base['recovery'] += 0.15
            base['mean_rev'] -= 0.10
            base['defensive'] -= 0.05
        
        # Below SMA200 -> more defensive
        if sma_dist < -0.05:
            base['defensive'] += 0.10
            base['trend'] -= 0.10
        
        # Normalize weights to sum to 1
        total = sum(base.values())
        for k in base:
            base[k] = max(0.05, base[k] / total)  # Min 5% per agent
        
        # Re-normalize after clamping
        total = sum(base.values())
        return {k: v/total for k, v in base.items()}

    def run_backtest(self, start_date=None, end_date=None, plot_results=True):
        s_date = start_date if start_date else self.config['TEST_START']
        e_date = end_date if end_date else self.config['TEST_END']
        
        if plot_results: print(f"=== 2. RUNNING V9 BACKTEST ({s_date} to {e_date}) ===")
        
        warmup_dt = pd.Timestamp(s_date) - pd.Timedelta(days=365)
        full_data = self.dp.get_data(warmup_dt.strftime('%Y-%m-%d'), e_date)
        if full_data.empty: return 0.0
        
        test_indices = np.where((full_data.index >= s_date) & (full_data.index <= e_date))[0]
        if len(test_indices) == 0: return 0.0

        portfolio = self.config['INITIAL_BALANCE']
        benchmark_equity = self.config['INITIAL_BALANCE'] 
        holdings = 0
        history = []
        
        cols = full_data.columns.tolist()
        idx_adx = cols.index('adx')
        idx_sma = cols.index('dist_sma200')
        idx_ret = cols.index('returns')
        idx_vol_pct = cols.index('vol_percentile')
        idx_raw_vol = cols.index('realized_vol_20d')
        idx_vix = cols.index('vix_norm') if 'vix_norm' in cols else -1
        idx_momentum = cols.index('momentum_50d') if 'momentum_50d' in cols else -1
        idx_drawdown = cols.index('drawdown') if 'drawdown' in cols else -1
        
        data_vals = full_data.values
        dates = full_data.index
        window = self.config['WINDOW_SIZE']
        
        agent_contribs = {name: 0.0 for name in self.agents.keys()}
        
        if plot_results:
            print(f"{'Date':<12} | {'Weights':<40} | {'Ensemble':<10} | {'Portfolio':<10} | {'Bench':<10}")
            print("-" * 100)
        
        for t in test_indices:
            obs_raw = data_vals[t-window : t].flatten()
            
            # Get indicators for weight calculation
            indicators = {
                'adx': data_vals[t-1, idx_adx],
                'sma_dist': data_vals[t-1, idx_sma],
                'vix': data_vals[t-1, idx_vix] * 40 + 15 if idx_vix != -1 else 15,
                'momentum': data_vals[t-1, idx_momentum] if idx_momentum != -1 else 0,
                'drawdown': data_vals[t-1, idx_drawdown] if idx_drawdown != -1 else 0
            }
            
            # Calculate dynamic weights
            if self.config['USE_WEIGHTED_ENSEMBLE']:
                weights = self.calculate_weights(indicators)
            else:
                # Equal weights fallback
                weights = {k: 1/len(self.agents) for k in self.agents.keys()}
            
            # Get predictions from all agents
            agent_actions = {}
            for name, agent in self.agents.items():
                agent_actions[name] = agent.predict(obs_raw, deterministic=True)
            
            # Weighted ensemble action
            raw_action = sum(weights[k] * agent_actions[k] for k in agent_actions)
            
            # Execution
            mkt_ret = data_vals[t, idx_ret]
            raw_vol = data_vals[t-1, idx_raw_vol]
            sma_dist = indicators['sma_dist']
            
            # Vol Targeting
            vol_scaler = 1.0
            if self.config['USE_VOL_TARGETING'] and raw_vol > 0.01:
                vol_scaler = self.config['TARGET_VOL'] / raw_vol
            vol_scaler = np.clip(vol_scaler, 0.1, 2.0)
            
            scaled_action = raw_action * self.config['ACTION_SCALER'] * vol_scaler
            
            # Long only constraint
            if self.config['LONG_ONLY']:
                scaled_action = np.clip(scaled_action, 0, 10)
            
            # Position sizing
            position_size = np.clip(scaled_action, -self.config['MAX_LEVERAGE'], self.config['MAX_LEVERAGE'])
            
            # PnL
            cost = abs(position_size - holdings) * self.config['FEES']
            lev_cost = max(0, abs(position_size)-1) * self.config['BORROW_RATE']
            
            step_pnl_pct = (position_size * mkt_ret) - cost - lev_cost
            step_pnl_dollars = portfolio * step_pnl_pct
            
            portfolio *= (1 + step_pnl_pct)
            benchmark_equity *= (1 + mkt_ret)
            holdings = position_size
            
            # Track agent contributions
            for name in agent_actions:
                agent_contribs[name] += weights[name] * step_pnl_dollars
                
            if plot_results and t % 20 == 0:
                weights_str = " ".join([f"{k[:3]}:{v:.2f}" for k, v in weights.items()])
                print(f"{str(dates[t].date()):<12} | {weights_str:<40} | {position_size:<10.2f} | {portfolio:<10.0f} | {benchmark_equity:<10.0f}")
                
            history.append({
                'Date': dates[t],
                'Portfolio': portfolio,
                'Weights': weights.copy(),
                'Return': step_pnl_pct,
                'Benchmark': mkt_ret,
                'Position': position_size,
            })
            
        res = pd.DataFrame(history).set_index('Date')
        total_ret = (portfolio / self.config['INITIAL_BALANCE']) - 1
        
        if plot_results:
            bench_ret = (1 + res['Benchmark']).cumprod().iloc[-1] - 1
            print(f"\n=== V9 FINAL REPORT: {total_ret:.2%} (Bench: {bench_ret:.2%}) ===")
            print(f"Alpha: {(total_ret - bench_ret):.2%}")
            
            print("\n=== AGENT CONTRIBUTION BREAKDOWN ===")
            for name, contrib in agent_contribs.items():
                print(f"  {name:<12}: ${contrib:>10,.0f}")
            
            self.plot_dashboard(res)
            
        return total_ret
    
    def plot_dashboard(self, res):
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2]})
        plt.subplots_adjust(hspace=0.2)
        
        res['Bench_Equity'] = (1 + res['Benchmark']).cumprod() * self.config['INITIAL_BALANCE']
        
        ax0 = axes[0]
        ax0.plot(res.index, res['Portfolio'], label='V9 Ensemble', color='black', linewidth=2)
        ax0.plot(res.index, res['Bench_Equity'], label='Buy & Hold', color='gray', linestyle='--', alpha=0.6)
        ax0.set_title(f"V9 Equity Curve: {self.config['TARGET_ASSET']}", fontweight='bold')
        ax0.set_ylabel("Portfolio Value ($)")
        ax0.legend(loc='upper left')
        ax0.grid(True, alpha=0.3)

        ax1 = axes[1]
        strat_peak = res['Portfolio'].cummax()
        strat_dd = (res['Portfolio'] - strat_peak) / strat_peak
        ax1.fill_between(res.index, strat_dd, 0, color='red', alpha=0.3, label='Drawdown')
        ax1.set_ylabel("Drawdown %")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[2]
        colors = ['forestgreen' if r > 0 else 'firebrick' for r in res['Return']]
        ax2.bar(res.index, res['Position'], color=colors, width=1.5)
        ax2.set_ylabel("Position Size")
        ax2.grid(True, alpha=0.3)
        
        if self.config.get('SAVE_PLOTS', False):
            plot_path = os.path.join(
                self.config.get('PLOTS_DIR', './plots'), 
                f"v9_backtest_{self.config['TARGET_ASSET']}.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        plt.close(fig)


def run_system(tickers, target, start, end, optimize=False):
    config = DEFAULT_CONFIG.copy()
    config['TICKERS'] = tickers
    config['TARGET_ASSET'] = target
    config['TEST_START'] = start
    config['TEST_END'] = end
    
    # Safety defaults for single stocks
    if target != "SPY":
        config['LONG_ONLY'] = True
        config['MAX_LEVERAGE'] = 1.0
        config['TRAINING_STEPS'] = 30000
        config['USE_VOL_TARGETING'] = False
        config['ACTION_SCALER'] = 10.0
    
    print(f"\n>>> INITIALIZING V9 (4-Agent Seed Ensemble) <<<")
    print(f"Seeds: {config['SEEDS']}")
    print(f"Agents: trend, mean_rev, defensive, recovery")
    
    mgr = EnsembleManager(config)
    mgr.train_specialists()
    mgr.run_backtest()


if __name__ == "__main__":
    MY_TICKERS = ["GOOG", "^VIX", "SHY"]
    MY_TARGET = "GOOG"
    START = "2024-01-22"
    END = "2025-05-05"

    run_system(MY_TICKERS, MY_TARGET, START, END)
