import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import itertools
import copy

# ==========================================
# 1. DEFAULT CONFIGURATION
# ==========================================
DEFAULT_CONFIG = {
    # Assets
    "TICKERS": ["SPY", "IWM", "^VIX", "SHY"], 
    "TARGET_ASSET": "SPY",
    
    # Dates
    "TRAIN_START": "2015-01-01",
    "TRAIN_END": "2023-12-31",
    "TEST_START": "2024-01-02",
    "TEST_END": "2025-01-01",
    
    # Crisis Training Periods
    # MODIFICATION: Tightened periods to focus on the DRAWDOWN phase, not the recovery
    "CRISIS_PERIODS": [
        ("2018-10-01", "2018-12-24"), # Volmageddon (Drop only)
        # ("2020-01-01", "2020-05-01"), # OLD (Included recovery)
        ("2020-02-15", "2020-03-23"), # COVID (Crash only)
        ("2022-01-01", "2022-10-14"), # Bear (Until bottom)
    ],
    
    # Hyperparameters
    "WINDOW_SIZE": 20, 
    "ADX_THRESHOLD": 25,
    "TARGET_VOL": 0.15, # MODIFICATION: Lowered from 0.20 to be more conservative
    
    # Fixed Params
    "BB_STD": 2.0,
    "LEARNING_RATE": 3e-4,
    "BATCH_SIZE": 64,
    "TRAINING_STEPS": 50000, 
    "INITIAL_BALANCE": 100000,
    "FEES": 0.0005,
    "BORROW_RATE": 0.0002, 
    "ACTION_SCALER": 3.0, # MODIFICATION: Lowered from 5.0 to reduce over-confidence
    "MAX_LEVERAGE": 1.0, 
    "USE_VOL_TARGETING": True,
    "SMA_TREND_FILTER": True,
    "LONG_ONLY": False 
}

# ==========================================
# 2. DATA PROCESSOR
# ==========================================
class DataProcessor:
    def __init__(self, tickers, config):
        self.tickers = list(set(tickers))
        self.config = config
        
    def download(self, start, end):
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Safety check
                if self.config['TARGET_ASSET'] not in self.tickers:
                    self.tickers.append(self.config['TARGET_ASSET'])
                    
                print(f"Fetching data for: {self.tickers} ({start} to {end}) [Attempt {attempt+1}/{max_retries}]")
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
                
                # Check for critical emptiness
                if data.empty or ohlc.empty:
                    raise ValueError("Downloaded data is empty")
                    
                return data, ohlc
            except Exception as e:
                print(f"Data Download Error (Attempt {attempt+1}): {e}")
                time.sleep(2 * (attempt + 1)) # Backoff
                
        print("CRITICAL: Failed to download data after retries.")
        return pd.DataFrame(), pd.DataFrame()

    def add_features(self, df, ohlc):
        target = self.config['TARGET_ASSET']
        if df.empty or target not in df.columns: return pd.DataFrame()
        
        df = df.copy()
        
        # 1. Base Returns
        df['returns'] = df[target].pct_change()
        
        # 2. Trend (ADX)
        df['tr'] = np.maximum(ohlc['High'] - ohlc['Low'], 
                   np.maximum(abs(ohlc['High'] - ohlc['Close'].shift(1)), 
                              abs(ohlc['Low'] - ohlc['Close'].shift(1))))
        df['dm_plus'] = np.where((ohlc['High'] - ohlc['High'].shift(1)) > (ohlc['Low'].shift(1) - ohlc['Low']), 
                                 np.maximum(ohlc['High'] - ohlc['High'].shift(1), 0), 0)
        df['dm_minus'] = np.where((ohlc['Low'].shift(1) - ohlc['Low']) > (ohlc['High'] - ohlc['High'].shift(1)), 
                                  np.maximum(ohlc['Low'].shift(1) - ohlc['Low'], 0), 0)
        
        window = 14
        df['tr_s'] = df['tr'].rolling(window).mean()
        df['dp_s'] = df['dm_plus'].rolling(window).mean()
        df['dm_s'] = df['dm_minus'].rolling(window).mean()
        
        df['di_plus'] = 100 * (df['dp_s'] / df['tr_s'])
        df['di_minus'] = 100 * (df['dm_s'] / df['tr_s'])
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].rolling(window).mean()
        
        # 3. Mean Reversion
        sma = df[target].rolling(20).mean()
        std = df[target].rolling(20).std()
        df['bb_width'] = (std * 2 * 2) / sma
        df['pct_b'] = (df[target] - (sma - 2*std)) / (4*std)
        
        # RSI
        up = df['returns'].clip(lower=0)
        down = -1 * df['returns'].clip(upper=0)
        ma_up = up.rolling(window).mean()
        ma_down = down.rolling(window).mean()
        rs = ma_up / (ma_down + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 4. Crisis
        if '^VIX' in df.columns:
            df['vix_norm'] = (df['^VIX'] - 15) / 40
        else:
            df['vix_norm'] = 0
            
        sma200 = df[target].rolling(200).mean()
        df['dist_sma200'] = (df[target] - sma200) / sma200
        
        # 5. Volatility
        df['realized_vol_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['vol_percentile'] = df['realized_vol_20d'].rolling(252).rank(pct=True)
        
        return df.fillna(0)

    def get_data(self, start, end):
        df, ohlc = self.download(start, end)
        return self.add_features(df, ohlc)

    def get_crisis_data(self):
        dfs = []
        for s, e in self.config['CRISIS_PERIODS']:
            d, o = self.download(s, e)
            dfs.append(self.add_features(d, o))
        return pd.concat(dfs).reset_index(drop=True).fillna(0)

# ==========================================
# 3. UNIFIED AGENT ENVIRONMENT
# ==========================================
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
        
    def reset(self, seed=None, options=None):
        self.current_step = self.window
        return self._get_obs(), {}
    
    def _get_obs(self):
        return self.data[self.current_step-self.window : self.current_step].flatten()

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0, True, False, {}
            
        act = np.clip(action[0], -1, 1)
        ret = self.data[self.current_step, 0] 
        
        # MODIFICATION: Refined Reward Functions
        reward = 0
        
        if self.mode == 'trend':
            reward = act * ret * 100
            # Trend Encouragement
            if self.idx_sma != -1:
                sma_dist = self.data[self.current_step, self.idx_sma]
                if sma_dist > 0 and act > 0.1: reward += 0.05 * act 
                elif sma_dist < 0 and act < -0.1: reward += 0.05 * abs(act)
                
        elif self.mode == 'mean_rev':
            # Add transaction penalty to reduce noise
            reward = (act * ret * 100) - (0.05 * abs(act))
            
        elif self.mode == 'crisis':
            # CRISIS AGENT: SHOULD BE DEFENSIVE
            # If market is down, making money (shorting) is good. 
            # If market is up, being neutral is bettter than being long (risk averse).
            
            pnl = act * ret * 100
            
            # 1. Base PnL
            reward = pnl
            
            # 2. Penalty for Long Exposure in Crisis (Moderate)
            if act > 0: 
                reward -= (act * 0.5) # Tax on being long
            
            # 3. Bonus for "Survival" (Near zero exposure is okay)
            if abs(act) < 0.2:
                reward += 0.05
                
            # 4. Asymmetric Downside Protection (Heavily punish big losses)
            if pnl < -1.0: # Lost more than 1% in a step
                reward *= 2.0 # Amplify the penalty
            
        self.current_step += 1
        return self._get_obs(), reward, False, False, {}

# ==========================================
# 4. ENSEMBLE MANAGER
# ==========================================
class EnsembleManager:
    def __init__(self, config):
        self.config = config
        self.dp = DataProcessor(config['TICKERS'], config)
        self.agents = {}
        self.envs = {}
        
    def train_specialists(self, verbose=True):
        if verbose: print("\n=== 1. TRAINING SPECIALIST AGENTS ===")
        trend_data = self.dp.get_data(self.config['TRAIN_START'], self.config['TRAIN_END'])
        
        # 1. Trend
        env_trend = DummyVecEnv([lambda: TradingEnv(trend_data, self.config, mode='trend')])
        env_trend = VecNormalize(env_trend, norm_obs=True, norm_reward=False)
        model_trend = PPO("MlpPolicy", env_trend, verbose=0, learning_rate=self.config['LEARNING_RATE'])
        model_trend.learn(total_timesteps=self.config['TRAINING_STEPS'])
        self.agents['trend'] = model_trend
        self.envs['trend'] = env_trend
        
        # 2. Mean Rev
        env_mr = DummyVecEnv([lambda: TradingEnv(trend_data, self.config, mode='mean_rev')]) 
        env_mr = VecNormalize(env_mr, norm_obs=True, norm_reward=False)
        model_mr = PPO("MlpPolicy", env_mr, verbose=0, learning_rate=self.config['LEARNING_RATE'])
        # Fewer steps for mean rev? Or same.
        model_mr.learn(total_timesteps=self.config['TRAINING_STEPS'])
        self.agents['mean_rev'] = model_mr
        self.envs['mean_rev'] = env_mr
        
        # 3. Crisis
        crash_data = self.dp.get_crisis_data()
        env_crisis = DummyVecEnv([lambda: TradingEnv(crash_data, self.config, mode='crisis')])
        env_crisis = VecNormalize(env_crisis, norm_obs=True, norm_reward=False)
        model_crisis = PPO("MlpPolicy", env_crisis, verbose=0, learning_rate=self.config['LEARNING_RATE'])
        model_crisis.learn(total_timesteps=self.config['TRAINING_STEPS'])
        self.agents['crisis'] = model_crisis
        self.envs['crisis'] = env_crisis
        
        if verbose: print("=== TRAINING COMPLETE ===\n")

    def run_backtest(self, start_date=None, end_date=None, plot_results=True):
        s_date = start_date if start_date else self.config['TEST_START']
        e_date = end_date if end_date else self.config['TEST_END']
        
        if plot_results: print(f"=== 2. RUNNING BACKTEST ({s_date} to {e_date}) ===")
        
        # Warmup fetch
        warmup_dt = pd.Timestamp(s_date) - pd.Timedelta(days=365)
        full_data = self.dp.get_data(warmup_dt.strftime('%Y-%m-%d'), e_date)
        if full_data.empty: return 0.0
        
        test_indices = np.where((full_data.index >= s_date) & (full_data.index <= e_date))[0]
        if len(test_indices) == 0: return 0.0

        # Execution Loops
        portfolio = self.config['INITIAL_BALANCE']
        holdings = 0
        history = []
        
        # Pre-calc column indices
        cols = full_data.columns.tolist()
        idx_adx = cols.index('adx')
        idx_sma = cols.index('dist_sma200')
        idx_ret = cols.index('returns')
        idx_vol_pct = cols.index('vol_percentile')
        idx_raw_vol = cols.index('realized_vol_20d')
        
        data_vals = full_data.values
        dates = full_data.index
        window = self.config['WINDOW_SIZE']
        
        agent_pnls = {'trend': 0.0, 'mean_rev': 0.0, 'crisis': 0.0}
        
        # Rolling Regime Tracker for Stability
        regime_history = []
        
        if plot_results:
            print(f"{'Date':<12} | {'Regime':<10} | {'Active Agent':<10} | {'Raw Act':<8} | {'Scale Act':<10} | {'Balance':<10}")
            print("-" * 90)
        
        for t in test_indices:
            obs_raw = data_vals[t-window : t].flatten()
            
            # Regime Logic
            vol_pct = data_vals[t-1, idx_vol_pct] 
            raw_vol = data_vals[t-1, idx_raw_vol]
            adx = data_vals[t-1, idx_adx]
            sma_dist = data_vals[t-1, idx_sma]
            
            # MODIFICATION: Stricter/Stickier Regime Logic
            # 1. Hard Bear Rule
            # Relaxed for TSLA: -10% instead of -5%
            if sma_dist < -0.10: 
                 regime, active_agent_name = "CRISIS", "crisis"
            # 2. Vol Rule
            elif vol_pct > 0.90 and sma_dist < -0.05:
                 regime, active_agent_name = "CRISIS", "crisis"
            # 3. Trend Rule (Strong Trends Only)
            elif adx > 25 and sma_dist > 0: 
                regime, active_agent_name = "TREND", "trend"
            # 4. Default
            else:
                regime, active_agent_name = "CHOP", "mean_rev"
                
            # Smoothing (must hold regime for at least 3 days to switch, unless Crisis)
            if len(regime_history) > 3:
                last_3 = regime_history[-3:]
                
                # MODIFICATION: Allow breaking OUT of Mean Rev if Trend is strong
                if regime == 'TREND' and last_3[-1] == 'mean_rev':
                    pass # Allow switch
                
                # Allow breaking OUT of Crisis if the signal is distinct
                elif regime == 'TREND' and last_3[-1] == 'crisis':
                     pass # Allow immediate switch to Trend
                     
                elif regime != 'CRISIS' and active_agent_name != last_3[-1]:
                    # resist switch
                    if last_3.count(last_3[-1]) == 3:
                        # Keep previous
                        active_agent_name = last_3[-1]
                        regime = "HELD"
            
            regime_history.append(active_agent_name)
                
            # Predict
            agent = self.agents[active_agent_name]
            norm_env = self.envs[active_agent_name]
            obs_norm = norm_env.normalize_obs(obs_raw)
            action, _ = agent.predict(obs_norm, deterministic=True)
            
            # Exec
            mkt_ret = data_vals[t, idx_ret]
            raw_action = action[0]
            
            # Volatility Targeting
            vol_scaler = 1.0
            if self.config['USE_VOL_TARGETING'] and raw_vol > 0.01:
                vol_scaler = self.config['TARGET_VOL'] / raw_vol
            vol_scaler = np.clip(vol_scaler, 0.1, 2.0)
            
            # Scaling
            # MODIFICATION: Dynamic Confidence Multiplier? 
            # Sticking to fixed but lower scaler for now.
            scaled_action = raw_action * self.config['ACTION_SCALER'] * vol_scaler
            
            if self.config['LONG_ONLY']: scaled_action = np.clip(scaled_action, 0, 10)
            
            # Trend Filter (Hard Override)
            if self.config['SMA_TREND_FILTER'] and active_agent_name == 'trend':
                if sma_dist < -0.03: scaled_action = np.clip(scaled_action, -10, 0) # Tolerant to 3% dip
                elif sma_dist > 0: scaled_action = np.clip(scaled_action, 0, 10)
            
            position_size = np.clip(scaled_action, -self.config['MAX_LEVERAGE'], self.config['MAX_LEVERAGE'])
            
            # MODIFICATION: Hard Guardrail for Crisis Safety
            # If in Crisis, DO NOT allow Long positions (Cash or Short only)
            if active_agent_name == 'crisis':
                position_size = np.clip(position_size, -1.0, 0.0)
            
            # PnL
            cost = abs(position_size - holdings) * self.config['FEES']
            lev_cost = max(0, abs(position_size)-1) * self.config['BORROW_RATE']
            
            step_pnl_pct = (position_size * mkt_ret) - cost - lev_cost
            step_pnl_dollars = portfolio * step_pnl_pct
            
            portfolio *= (1 + step_pnl_pct)
            holdings = position_size
            
            # Attribute PnL
            agent_pnls[active_agent_name] += step_pnl_dollars
            
            if plot_results and t % 20 == 0: 
                print(f"{str(dates[t].date()):<12} | {regime:<10} | {active_agent_name:<10} | {raw_action:<8.2f} | {position_size:<10.2f} | {portfolio:<10.0f}")
                
            history.append({
                'Date': dates[t],
                'Portfolio': portfolio,
                'Regime': regime,
                'Agent': active_agent_name,
                'Return': step_pnl_pct,
                'Benchmark': mkt_ret,
                'Scale Act': position_size,
                'Trend_PnL': agent_pnls['trend'],
                'MeanRev_PnL': agent_pnls['mean_rev'],
                'Crisis_PnL': agent_pnls['crisis']
            })
            
        res = pd.DataFrame(history).set_index('Date')
        total_ret = (portfolio / self.config['INITIAL_BALANCE']) - 1
        
        if plot_results:
            bench_ret = (1 + res['Benchmark']).cumprod().iloc[-1] - 1
            print(f"\n=== FINAL REPORT: {total_ret:.2%} (Bench: {bench_ret:.2%}) ===")
            self.plot_dashboard(res)
            
        return total_ret

    def plot_dashboard(self, res):
        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2, 2]})
        plt.subplots_adjust(hspace=0.2)
        
        # --- Panel 1: Equity & Regimes ---
        res['Bench_Equity'] = (1 + res['Benchmark']).cumprod() * self.config['INITIAL_BALANCE']
        ax0 = axes[0]
        ax0.plot(res.index, res['Portfolio'], label='Ensemble AI', color='black', linewidth=2)
        ax0.plot(res.index, res['Bench_Equity'], label='Buy & Hold', color='gray', linestyle='--', alpha=0.6)
        ax0.set_title(f"Equity Curve: {self.config['TARGET_ASSET']}", fontweight='bold')
        ax0.set_ylabel("Portfolio Value ($)")
        ax0.grid(True, alpha=0.3)
        
        # Regime Shading
        y_min, y_max = ax0.get_ylim()
        ax0.fill_between(res.index, y_min, y_max, where=(res['Agent'] == 'trend'), color='green', alpha=0.1, label='Trend Regime')
        ax0.fill_between(res.index, y_min, y_max, where=(res['Agent'] == 'mean_rev'), color='orange', alpha=0.15, label='Chop Regime')
        ax0.fill_between(res.index, y_min, y_max, where=(res['Agent'] == 'crisis'), color='red', alpha=0.15, label='Crisis Regime')
        ax0.legend(loc='upper left', frameon=True)

        # --- Panel 2: Underwater Plot (Drawdown) ---
        ax1 = axes[1]
        
        # Calculate Drawdowns
        strat_peak = res['Portfolio'].cummax()
        strat_dd = (res['Portfolio'] - strat_peak) / strat_peak
        
        bench_peak = res['Bench_Equity'].cummax()
        bench_dd = (res['Bench_Equity'] - bench_peak) / bench_peak
        
        ax1.fill_between(res.index, strat_dd, 0, color='red', alpha=0.3, label='Strategy Drawdown')
        ax1.plot(res.index, bench_dd, color='gray', linestyle='--', linewidth=1, label='Benchmark Drawdown')
        ax1.set_ylabel("Drawdown %")
        ax1.set_title("Risk Profile: Underwater Plot", fontsize=10)
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)

        # --- Panel 3: Agent Contribution (PnL Attribution) ---
        ax2 = axes[2]
        ax2.plot(res.index, res['Trend_PnL'], label='Trend Agent', color='green', linewidth=1.5)
        ax2.plot(res.index, res['MeanRev_PnL'], label='Mean Rev Agent', color='orange', linewidth=1.5)
        ax2.plot(res.index, res['Crisis_PnL'], label='Crisis Agent', color='red', linewidth=1.5)
        ax2.set_ylabel("Profit Contrib ($)")
        ax2.set_title("Strategy Attribution: Who made the money?", fontsize=10)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # --- Panel 4: Exposure Heatmap (Position Size + PnL) ---
        ax3 = axes[3]
        colors = ['forestgreen' if r > 0 else 'firebrick' for r in res['Return']]
        
        ax3.bar(res.index, res['Scale Act'], color=colors, width=1.5, label='Exposure (Color=PnL)')
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.set_ylabel("Exposure (0.0 - 1.0)")
        ax3.set_title("Conviction & Outcome (Green=Win, Red=Loss)", fontsize=10)
        ax3.set_ylim(-0.1, self.config['MAX_LEVERAGE']*1.1)
        ax3.grid(True, alpha=0.3)
        
        # Formatting
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.savefig(f"backtest_{self.config['TARGET_ASSET']}_v4.png", dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to: backtest_{self.config['TARGET_ASSET']}_v4.png")
        plt.show()

# ==========================================
# 5. HYPERPARAMETER OPTIMIZATION
# ==========================================
class HyperparameterOptimizer:
    def __init__(self, tickers, target):
        self.tickers = tickers
        self.target = target
        
    def optimize(self, train_start, train_end):
        print(f"\n>>> STARTING HYPERPARAMETER OPTIMIZATION FOR {self.target} <<<")
        
        # Grid Search Space
        param_grid = {
            "WINDOW_SIZE": [20, 60],
            "ADX_THRESHOLD": [20, 25],
            "TARGET_VOL": [0.15, 0.25] 
        }
        
        best_ret = -np.inf
        best_config = None
        
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for i, params in enumerate(combinations):
            print(f"Testing Config {i+1}/{len(combinations)}: {params}")
            
            # Create Config Copy
            trial_config = DEFAULT_CONFIG.copy()
            trial_config.update(params)
            trial_config['TICKERS'] = self.tickers
            trial_config['TARGET_ASSET'] = self.target
            trial_config['TRAIN_START'] = train_start
            trial_config['TRAIN_END'] = train_end
            
            # Validation
            trial_config['TEST_START'] = str(int(train_end[:4]) - 1) + train_end[4:]
            trial_config['TEST_END'] = train_end
            
            # Fast Training
            trial_config['TRAINING_STEPS'] = 10000 
            
            mgr = EnsembleManager(trial_config)
            mgr.train_specialists(verbose=False)
            ret = mgr.run_backtest(plot_results=False)
            
            print(f"  -> Return: {ret:.2%}")
            
            if ret > best_ret:
                best_ret = ret
                best_config = params
                
        print(f"\n>>> BEST PARAMETERS FOUND: {best_config} (Ret: {best_ret:.2%}) <<<")
        return best_config

def run_system(tickers, target, start, end, optimize=False):
    config = DEFAULT_CONFIG.copy()
    config['TICKERS'] = tickers
    config['TARGET_ASSET'] = target
    config['TEST_START'] = start
    config['TEST_END'] = end
    
    # Safety Defaults for Single Stock
    if target != "SPY":
        config['LONG_ONLY'] = True
        config['MAX_LEVERAGE'] = 1.0
        config['TARGET_VOL'] = 0.35 # Relaxed baseline
        config['ADX_THRESHOLD'] = 25 # Standard
        config['TRAINING_STEPS'] = 100000 
        
        # CRITICAL RESTORATION: Disable Vol Targeting for Single Stocks to capture full upside
        config['USE_VOL_TARGETING'] = False 
        config['ACTION_SCALER'] = 5.0 # Balanced
        
    if optimize:
        opt = HyperparameterOptimizer(tickers, target)
        best_params = opt.optimize("2015-01-01", "2023-12-31") 
        config.update(best_params)
    
    print("\n>>> INITIALIZING FINAL RUN (V4) <<<")
    print(f"Config: Window={config['WINDOW_SIZE']}, VolTarget={config['TARGET_VOL']}, ADX={config['ADX_THRESHOLD']}")
    
    mgr = EnsembleManager(config)
    mgr.train_specialists()
    mgr.run_backtest()

if __name__ == "__main__":
    # USER SETTINGS
    MY_TICKERS = ["TSLA", "^VIX", "SHY"]
    MY_TARGET = "TSLA"
    START = "2024-01-02"
    END = "2026-06-01" # Extended
    
    # Set optimize=True to find best params automatically
    # run_system(MY_TICKERS, MY_TARGET, START, END, optimize=True)
    
    # Run Standard
    run_system(MY_TICKERS, MY_TARGET, START, END, optimize=False)
