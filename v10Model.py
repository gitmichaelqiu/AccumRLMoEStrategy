"""
V10 Trading Model - Aggressive Hybrid

Key Improvements over V9:
1. Conviction multiplier when agents agree
2. Momentum override (from V8) with seed ensemble
3. Aggressive base weights (trend 45%, recovery 25%)
4. Reduced defensive drag
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- V10 CONFIGURATION ---
DEFAULT_CONFIG = {
    # Assets
    "TICKERS": ["SPY", "IWM", "^VIX", "SHY"], 
    "TARGET_ASSET": "SPY",
    
    # Dates
    "TRAIN_START": "2015-01-01",
    "TRAIN_END": "2023-12-31",
    "TEST_START": "2024-01-02",
    "TEST_END": "2025-05-05",
    
    # Crisis Training Periods
    "CRISIS_PERIODS": [
        ("2018-10-01", "2019-01-01"),
        ("2020-01-01", "2020-05-01"),
        ("2022-01-01", "2022-12-31"),
        ("2018-01-01", "2018-04-01"),
    ],
    
    # Recovery Training Periods
    "RECOVERY_PERIODS": [
        ("2019-01-01", "2019-06-01"),
        ("2020-04-01", "2020-09-01"),
        ("2023-01-01", "2023-07-01"),
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
    
    # V10 AGGRESSIVE: Base Weights (shifted to favor trend/recovery)
    "BASE_WEIGHTS": {
        'trend': 0.45,      # Was 0.35
        'mean_rev': 0.15,   # Was 0.25
        'defensive': 0.15,  # Was 0.25
        'recovery': 0.25    # Was 0.15
    },
    
    # V10 AGGRESSIVE: Conviction settings
    "CONVICTION_THRESHOLD": 0.6,   # Agent agreement threshold
    "CONVICTION_BOOST": 1.3,       # 30% boost when agents agree
    
    # V10 AGGRESSIVE: Momentum override
    "USE_MOMENTUM_OVERRIDE": True,
    "MOMENTUM_OVERRIDE_THRESHOLD": 0.15,
    
    # V10 AGGRESSIVE: Defensive reduction
    "VIX_DEFENSIVE_TRIGGER": 30,   # Only full defensive when VIX > 30
    
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
        
        # Recovery indicator
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
        return pd.concat(dfs).reset_index(drop=True).fillna(0) if dfs else pd.DataFrame()
    
    def get_recovery_data(self):
        dfs = []
        for s, e in self.config['RECOVERY_PERIODS']:
            d, o = self.download(s, e)
            if not d.empty:
                dfs.append(self.add_features(d, o))
        return pd.concat(dfs).reset_index(drop=True).fillna(0) if dfs else pd.DataFrame()

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
            reward = act * ret * 100
            if self.idx_sma != -1:
                sma_dist = self.data[self.current_step, self.idx_sma]
                if sma_dist > 0 and act > 0.1: reward += 0.05 * act 
                elif sma_dist < 0 and act < -0.1: reward += 0.05 * abs(act)
                
        elif self.mode == 'mean_rev':
            if self.idx_pct_b != -1:
                pct_b = self.data[self.current_step, self.idx_pct_b]
                if pct_b < 0.3 and act > 0.5:
                    reward = act * ret * 150
                elif pct_b > 0.7 and act < 0:
                    reward = abs(act) * (-ret) * 100
                else:
                    reward = act * ret * 50
            else:
                reward = act * ret * 100

        elif self.mode == 'defensive':
            reward = act * ret * 100
            if ret < -0.01 and act < 0.1:
                reward += 0.5 * (1 - act)
            elif ret < -0.02 and act > 0.5:
                reward -= act * 20
            
        elif self.mode == 'recovery':
            reward = act * ret * 100
            if act > 0.7:
                reward += 0.1 * act
            if ret > 0.01 and act > 0.5:
                reward *= 1.5
            
        self.current_step += 1
        return self._get_obs(), reward, False, False, {}

# --- SEED ENSEMBLE AGENT ---
class SeedEnsembleAgent:
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
        actions = []
        for model, env in zip(self.models, self.envs):
            obs_norm = env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=deterministic)
            actions.append(action[0])
        return np.mean(actions)
    
    def get_env(self):
        return self.envs[0]

# --- V10 ENSEMBLE MANAGER ---
class EnsembleManager:
    def __init__(self, config):
        self.config = config
        self.dp = DataProcessor(config['TICKERS'], config)
        self.agents = {}
        
        if config.get('SAVE_PLOTS', False):
            os.makedirs(config.get('PLOTS_DIR', './plots'), exist_ok=True)
        
    def train_specialists(self, verbose=True):
        if verbose: print("\n=== 1. TRAINING V10 AGGRESSIVE AGENTS ===")
        
        trend_data = self.dp.get_data(self.config['TRAIN_START'], self.config['TRAIN_END'])
        crisis_data = self.dp.get_crisis_data()
        recovery_data = self.dp.get_recovery_data()
        
        seeds = self.config['SEEDS'] if self.config['USE_SEED_ENSEMBLE'] else [42]
        
        if verbose: print(f"  Training TREND agent with seeds {seeds}...")
        self.agents['trend'] = SeedEnsembleAgent(
            env_fn=lambda: TradingEnv(trend_data, self.config, mode='trend'),
            config=self.config, seeds=seeds, mode='trend'
        )
        
        if verbose: print(f"  Training MEAN_REV agent with seeds {seeds}...")
        self.agents['mean_rev'] = SeedEnsembleAgent(
            env_fn=lambda: TradingEnv(trend_data, self.config, mode='mean_rev'),
            config=self.config, seeds=seeds, mode='mean_rev'
        )
        
        if not crisis_data.empty:
            if verbose: print(f"  Training DEFENSIVE agent with seeds {seeds}...")
            self.agents['defensive'] = SeedEnsembleAgent(
                env_fn=lambda: TradingEnv(crisis_data, self.config, mode='defensive'),
                config=self.config, seeds=seeds, mode='defensive'
            )
        else:
            self.agents['defensive'] = self.agents['trend']
        
        if not recovery_data.empty:
            if verbose: print(f"  Training RECOVERY agent with seeds {seeds}...")
            self.agents['recovery'] = SeedEnsembleAgent(
                env_fn=lambda: TradingEnv(recovery_data, self.config, mode='recovery'),
                config=self.config, seeds=seeds, mode='recovery'
            )
        else:
            self.agents['recovery'] = self.agents['trend']
        
        if verbose: print("=== TRAINING COMPLETE ===\n")

    def calculate_weights(self, indicators):
        """V10: Aggressive weight calculation with reduced defensive drag"""
        adx = indicators.get('adx', 20)
        vix = indicators.get('vix', 15)
        momentum = indicators.get('momentum', 0)
        sma_dist = indicators.get('sma_dist', 0)
        
        base = self.config['BASE_WEIGHTS'].copy()
        
        # V10: Reduce defensive weight when VIX is low
        if vix < self.config['VIX_DEFENSIVE_TRIGGER']:
            base['defensive'] *= 0.5
            base['trend'] += 0.05
            base['recovery'] += 0.025
        
        # Strong trend -> more trend weight
        if adx > 30:
            base['trend'] += 0.15
            base['mean_rev'] -= 0.10
        
        # High VIX -> increase defensive (only if really high)
        if vix > 35:
            base['defensive'] += 0.15
            base['trend'] -= 0.10
        
        # Strong positive momentum -> boost trend/recovery
        if momentum > 0.10:
            base['trend'] += 0.10
            base['recovery'] += 0.10
            base['defensive'] -= 0.10
            base['mean_rev'] -= 0.10
        
        # Normalize
        total = sum(base.values())
        for k in base:
            base[k] = max(0.05, base[k] / total)
        total = sum(base.values())
        return {k: v/total for k, v in base.items()}

    def run_backtest(self, start_date=None, end_date=None, plot_results=True):
        s_date = start_date if start_date else self.config['TEST_START']
        e_date = end_date if end_date else self.config['TEST_END']
        
        if plot_results: print(f"=== 2. RUNNING V10 BACKTEST ({s_date} to {e_date}) ===")
        
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
        idx_raw_vol = cols.index('realized_vol_20d')
        idx_vix = cols.index('vix_norm') if 'vix_norm' in cols else -1
        idx_momentum = cols.index('momentum_50d') if 'momentum_50d' in cols else -1
        
        data_vals = full_data.values
        dates = full_data.index
        window = self.config['WINDOW_SIZE']
        
        momentum_override_count = 0
        conviction_boost_count = 0
        
        if plot_results:
            print(f"{'Date':<12} | {'Override':<8} | {'Position':<10} | {'Portfolio':<10} | {'Bench':<10}")
            print("-" * 70)
        
        for t in test_indices:
            obs_raw = data_vals[t-window : t].flatten()
            
            indicators = {
                'adx': data_vals[t-1, idx_adx],
                'sma_dist': data_vals[t-1, idx_sma],
                'vix': data_vals[t-1, idx_vix] * 40 + 15 if idx_vix != -1 else 15,
                'momentum': data_vals[t-1, idx_momentum] if idx_momentum != -1 else 0,
            }
            
            momentum = indicators['momentum']
            
            # V10: MOMENTUM OVERRIDE (like V8)
            if self.config['USE_MOMENTUM_OVERRIDE'] and momentum > self.config['MOMENTUM_OVERRIDE_THRESHOLD']:
                # Strong momentum -> full exposure, bypass ensemble
                final_action = 1.0
                override_flag = "MOM"
                momentum_override_count += 1
            else:
                # Normal ensemble voting
                weights = self.calculate_weights(indicators)
                
                agent_actions = {}
                for name, agent in self.agents.items():
                    agent_actions[name] = agent.predict(obs_raw, deterministic=True)
                
                # Weighted ensemble
                raw_action = sum(weights[k] * agent_actions[k] for k in agent_actions)
                
                # V10: CONVICTION MULTIPLIER
                conviction_thresh = self.config['CONVICTION_THRESHOLD']
                if all(a > conviction_thresh for a in agent_actions.values()):
                    raw_action *= self.config['CONVICTION_BOOST']
                    override_flag = "CNV"
                    conviction_boost_count += 1
                else:
                    override_flag = "-"
                
                final_action = raw_action
            
            # Execution
            mkt_ret = data_vals[t, idx_ret]
            raw_vol = data_vals[t-1, idx_raw_vol]
            
            # Vol Targeting
            vol_scaler = 1.0
            if self.config['USE_VOL_TARGETING'] and raw_vol > 0.01:
                vol_scaler = self.config['TARGET_VOL'] / raw_vol
            vol_scaler = np.clip(vol_scaler, 0.1, 2.0)
            
            scaled_action = final_action * self.config['ACTION_SCALER'] * vol_scaler
            
            if self.config['LONG_ONLY']:
                scaled_action = np.clip(scaled_action, 0, 10)
            
            position_size = np.clip(scaled_action, -self.config['MAX_LEVERAGE'], self.config['MAX_LEVERAGE'])
            
            # PnL
            cost = abs(position_size - holdings) * self.config['FEES']
            lev_cost = max(0, abs(position_size)-1) * self.config['BORROW_RATE']
            
            step_pnl_pct = (position_size * mkt_ret) - cost - lev_cost
            
            portfolio *= (1 + step_pnl_pct)
            benchmark_equity *= (1 + mkt_ret)
            holdings = position_size
                
            if plot_results and t % 20 == 0:
                print(f"{str(dates[t].date()):<12} | {override_flag:<8} | {position_size:<10.2f} | {portfolio:<10.0f} | {benchmark_equity:<10.0f}")
                
            history.append({
                'Date': dates[t],
                'Portfolio': portfolio,
                'Return': step_pnl_pct,
                'Benchmark': mkt_ret,
                'Position': position_size,
            })
            
        res = pd.DataFrame(history).set_index('Date')
        total_ret = (portfolio / self.config['INITIAL_BALANCE']) - 1
        
        if plot_results:
            bench_ret = (1 + res['Benchmark']).cumprod().iloc[-1] - 1
            print(f"\n=== V10 FINAL REPORT: {total_ret:.2%} (Bench: {bench_ret:.2%}) ===")
            print(f"Alpha: {(total_ret - bench_ret):.2%}")
            print(f"Momentum Overrides: {momentum_override_count}")
            print(f"Conviction Boosts: {conviction_boost_count}")
            
            self.plot_dashboard(res)
            
        return total_ret
    
    def plot_dashboard(self, res):
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2]})
        plt.subplots_adjust(hspace=0.2)
        
        res['Bench_Equity'] = (1 + res['Benchmark']).cumprod() * self.config['INITIAL_BALANCE']
        
        ax0 = axes[0]
        ax0.plot(res.index, res['Portfolio'], label='V10 Aggressive', color='blue', linewidth=2)
        ax0.plot(res.index, res['Bench_Equity'], label='Buy & Hold', color='gray', linestyle='--', alpha=0.6)
        ax0.set_title(f"V10 Equity Curve: {self.config['TARGET_ASSET']}", fontweight='bold')
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
                f"v10_backtest_{self.config['TARGET_ASSET']}.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        plt.close(fig)


if __name__ == "__main__":
    from v10Model import DEFAULT_CONFIG, EnsembleManager
    
    config = DEFAULT_CONFIG.copy()
    config['TICKERS'] = ["GOOG", "^VIX", "SHY"]
    config['TARGET_ASSET'] = "GOOG"
    config['TEST_START'] = "2024-01-02"
    config['TEST_END'] = "2025-02-05"
    config['LONG_ONLY'] = True
    config['TRAINING_STEPS'] = 30000
    
    print("\n>>> INITIALIZING V10 AGGRESSIVE MODEL <<<")
    mgr = EnsembleManager(config)
    mgr.train_specialists()
    mgr.run_backtest()
