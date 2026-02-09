"""
V13 Mixture of Experts Trading Model

Key Features:
1. GMM-based Soft Gating (4 regimes: Growth, Stagnation, Crisis, Transition)
2. Transaction Cost Penalty for smooth weight transitions
3. Conflict Detection - reduce position when agents disagree
4. Priority Override Hierarchy: Risk > Jump > Momentum > Soft MoE
5. Reusable modular components

Based on: "Adaptive Multi-Agent Architectures for Regime-Aware Quantitative Trading"
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

from data_utils import create_session, download_with_ohlc
from feature_engineering import add_technical_features, get_regime_features
from regime_detector import RegimeDetector, create_regime_features_df
from agents import TradingEnv, SeedEnsembleAgent, set_seed

warnings.filterwarnings('ignore')

# --- V13 CONFIGURATION ---
DEFAULT_CONFIG = {
    # Assets
    "TICKERS": ["SPY", "^VIX", "SHY"],
    "TARGET_ASSET": "SPY",
    
    # Dates
    "TRAIN_START": "2015-01-01",
    "TRAIN_END": "2023-12-31",
    "TEST_START": "2024-01-02",
    "TEST_END": "2025-02-05",
    
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
    
    # Core Hyperparameters
    "WINDOW_SIZE": 60,
    "LEARNING_RATE": 3e-4,
    "BATCH_SIZE": 64,
    "TRAINING_STEPS": 50000,
    "INITIAL_BALANCE": 100000,
    "FEES": 0.0005,
    "BORROW_RATE": 0.0002,
    "ACTION_SCALER": 10.0,  # More aggressive position sizing
    "MAX_LEVERAGE": 1.0,
    "LONG_ONLY": False,
    
    # V13: MoE Settings
    "N_REGIMES": 4,
    "REGIME_FEATURES": ["returns", "realized_vol_20d", "vix_norm", "momentum_50d"],
    
    # V13: Transaction Cost Penalty (λ|W_t - W_{t-1}|)
    "SWITCHING_COST_LAMBDA": 0.01,  # Reduced - allow faster transitions
    
    # V13: Conflict Detection (tuned to be less restrictive)
    "CONFLICT_THRESHOLD": 0.5,  # Higher threshold - only reduce on major disagreements
    "CONFLICT_PENALTY": 0.15,   # Reduced penalty when conflict detected
    
    # Override Thresholds
    "JUMP_THRESHOLD": 3.5,
    "MOMENTUM_LOOKBACK": 50,
    "MOMENTUM_OVERRIDE_THRESHOLD": 0.15,
    
    # Vol Targeting
    "USE_VOL_TARGETING": True,
    "TARGET_VOL": 0.40,
    
    # Risk Management (disable for single-stock focus, enable for portfolio)
    "RISK_MANAGEMENT": False,
    "RISK_DRAWDOWN_TIER1": -0.15,  # Sell Half (less sensitive)
    "RISK_DRAWDOWN_TIER2": -0.25,  # Cash Out (less sensitive)
    
    # Output Settings
    "SAVE_PLOTS": True,
    "PLOTS_DIR": "./Plots",
    
    # Debug
    "VERBOSE": True,
    "PRINT_INTERVAL": 20,  # Print every N steps
}


class DataProcessor:
    """Data processor using modular utilities."""
    
    def __init__(self, tickers, config):
        self.tickers = list(set(tickers))
        self.config = config
        self.session = create_session()
        
    def get_data(self, start, end):
        """Get processed data with all features."""
        target = self.config['TARGET_ASSET']
        if target not in self.tickers:
            self.tickers.append(target)
            
        df, ohlc = download_with_ohlc(
            self.tickers, target, start, end, self.session
        )
        
        if df.empty:
            return pd.DataFrame()
            
        return add_technical_features(df, ohlc, self.config)
    
    def get_crisis_data(self):
        """Get combined data from crisis periods."""
        dfs = []
        for s, e in self.config['CRISIS_PERIODS']:
            df = self.get_data(s, e)
            if not df.empty:
                dfs.append(df)
        return pd.concat(dfs).reset_index(drop=True).fillna(0) if dfs else pd.DataFrame()
    
    def get_recovery_data(self):
        """Get combined data from recovery periods."""
        dfs = []
        for s, e in self.config['RECOVERY_PERIODS']:
            df = self.get_data(s, e)
            if not df.empty:
                dfs.append(df)
        return pd.concat(dfs).reset_index(drop=True).fillna(0) if dfs else pd.DataFrame()


class MoEEnsembleManager:
    """
    Mixture of Experts Ensemble Manager (V13).
    
    Implements:
    - GMM-based soft gating for regime detection
    - Transaction cost penalty for smooth transitions
    - Conflict detection for uncertainty signaling
    - Priority override hierarchy
    """
    
    def __init__(self, config):
        self.config = config
        self.dp = DataProcessor(config['TICKERS'], config)
        self.agents = {}
        self.regime_detector = RegimeDetector(
            n_components=config['N_REGIMES'],
            random_state=config['SEEDS'][0]
        )
        
        # State for switching cost calculation
        self.prev_weights = None
        
        if config.get('SAVE_PLOTS', False):
            os.makedirs(config.get('PLOTS_DIR', './Plots'), exist_ok=True)
    
    def train_specialists(self, verbose=True):
        """Train regime-specialized agents and fit regime detector."""
        if verbose:
            print("\n" + "=" * 60)
            print("V13 MoE TRAINING")
            print("=" * 60)
        
        # 1. Load training data
        all_data = self.dp.get_data(
            self.config['TRAIN_START'],
            self.config['TRAIN_END']
        )
        
        if all_data.empty:
            raise ValueError("Failed to load training data")
        
        # 2. Fit Regime Detector
        if verbose:
            print("\n[1/5] Fitting GMM Regime Detector...")
        regime_features = get_regime_features(all_data)
        self.regime_detector.fit(regime_features, verbose=verbose)
        
        # 3. Load specialized training data
        crisis_data = self.dp.get_crisis_data()
        recovery_data = self.dp.get_recovery_data()
        
        seeds = self.config['SEEDS'] if self.config['USE_SEED_ENSEMBLE'] else [42]
        
        # 4. Train specialist agents
        if verbose:
            print(f"\n[2/5] Training TREND agent (seeds: {seeds})...")
        self.agents['trend'] = SeedEnsembleAgent(
            env_fn=lambda: TradingEnv(all_data, self.config, mode='trend'),
            config=self.config, seeds=seeds, mode='trend', verbose=verbose
        )
        
        if verbose:
            print(f"\n[3/5] Training MEAN_REV agent (seeds: {seeds})...")
        self.agents['mean_rev'] = SeedEnsembleAgent(
            env_fn=lambda: TradingEnv(all_data, self.config, mode='mean_rev'),
            config=self.config, seeds=seeds, mode='mean_rev', verbose=verbose
        )
        
        if not crisis_data.empty:
            if verbose:
                print(f"\n[4/5] Training DEFENSIVE agent (seeds: {seeds})...")
            self.agents['defensive'] = SeedEnsembleAgent(
                env_fn=lambda: TradingEnv(crisis_data, self.config, mode='defensive'),
                config=self.config, seeds=seeds, mode='defensive', verbose=verbose
            )
        else:
            self.agents['defensive'] = self.agents['trend']
        
        if not recovery_data.empty:
            if verbose:
                print(f"\n[5/5] Training RECOVERY agent (seeds: {seeds})...")
            self.agents['recovery'] = SeedEnsembleAgent(
                env_fn=lambda: TradingEnv(recovery_data, self.config, mode='recovery'),
                config=self.config, seeds=seeds, mode='recovery', verbose=verbose
            )
        else:
            self.agents['recovery'] = self.agents['trend']
        
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)
    
    def _calculate_switching_cost(self, weights):
        """
        Calculate transaction cost penalty for weight changes.
        Returns the penalty and updates prev_weights.
        """
        if self.prev_weights is None:
            self.prev_weights = weights.copy()
            return 0.0
        
        # Sum of absolute weight changes
        weight_change = sum(
            abs(weights[k] - self.prev_weights.get(k, 0))
            for k in weights
        )
        
        penalty = self.config['SWITCHING_COST_LAMBDA'] * weight_change
        self.prev_weights = weights.copy()
        return penalty
    
    def _detect_conflict(self, obs):
        """
        Detect if agents are in conflict (high disagreement).
        Returns conflict_detected (bool), conflict_severity (float 0-1).
        """
        agent_actions = {}
        for name, agent in self.agents.items():
            agent_actions[name] = agent.predict(obs, deterministic=True)
        
        actions = list(agent_actions.values())
        std_dev = np.std(actions)
        
        # Normalize std to 0-1 range (max possible std for [-1,1] is ~1.0)
        conflict_severity = min(std_dev / self.config['CONFLICT_THRESHOLD'], 1.0)
        conflict_detected = std_dev > self.config['CONFLICT_THRESHOLD']
        
        return conflict_detected, conflict_severity, agent_actions
    
    def run_backtest(self, start_date=None, end_date=None, plot_results=True):
        """Run backtest with V13 MoE logic."""
        s_date = start_date or self.config['TEST_START']
        e_date = end_date or self.config['TEST_END']
        
        print(f"\n{'=' * 60}")
        print(f"V13 MoE BACKTEST: {s_date} to {e_date}")
        print(f"Target: {self.config['TARGET_ASSET']}")
        print(f"{'=' * 60}")
        
        # Load data with warmup
        warmup_dt = pd.Timestamp(s_date) - pd.Timedelta(days=365)
        full_data = self.dp.get_data(warmup_dt.strftime('%Y-%m-%d'), e_date)
        
        if full_data.empty:
            print("ERROR: No data available")
            return 0.0
        
        test_indices = np.where(
            (full_data.index >= s_date) & (full_data.index <= e_date)
        )[0]
        
        if len(test_indices) == 0:
            print("ERROR: No test data in range")
            return 0.0
        
        # Initialize
        portfolio = self.config['INITIAL_BALANCE']
        benchmark_equity = self.config['INITIAL_BALANCE']
        holdings = 0
        history = []
        self.prev_weights = None
        
        # Column indices
        cols = full_data.columns.tolist()
        idx_ret = cols.index('returns')
        idx_vol = cols.index('realized_vol_20d')
        idx_vix = cols.index('vix_norm') if 'vix_norm' in cols else -1
        idx_mom = cols.index('momentum_50d') if 'momentum_50d' in cols else -1
        idx_dd252 = cols.index('drawdown_252') if 'drawdown_252' in cols else -1
        idx_sma = cols.index('dist_sma200') if 'dist_sma200' in cols else -1
        idx_lstat = cols.index('l_stat') if 'l_stat' in cols else -1
        
        data_vals = full_data.values
        dates = full_data.index
        window = self.config['WINDOW_SIZE']
        
        # Counters for debugging
        override_counts = {
            'risk_max': 0, 'risk_mid': 0, 'jump_up': 0, 'jump_down': 0,
            'momentum': 0, 'conflict': 0, 'moe': 0
        }
        
        if self.config['VERBOSE']:
            print(f"\n{'Date':<12} | {'Override':<8} | {'Regime':<10} | {'Pos':<8} | {'Port':>10} | {'Bench':>10}")
            print("-" * 75)
        
        for t in test_indices:
            obs_raw = data_vals[t - window : t].flatten()
            
            # Current indicators
            ret_t = data_vals[t-1, idx_ret]
            vol_t = data_vals[t-1, idx_vol]
            vix_t = data_vals[t-1, idx_vix] if idx_vix != -1 else 0
            mom_t = data_vals[t-1, idx_mom] if idx_mom != -1 else 0
            dd252_t = data_vals[t-1, idx_dd252] if idx_dd252 != -1 else 0
            sma_dist_t = data_vals[t-1, idx_sma] if idx_sma != -1 else 0
            l_stat_t = data_vals[t-1, idx_lstat] if idx_lstat != -1 else 0
            
            override_flag = ""
            regime_name = "-"
            final_action = 0.0
            risk_override = False
            position_scale = 1.0
            
            # ============================================================
            # PRIORITY 1: RISK MANAGEMENT OVERRIDE
            # ============================================================
            if self.config.get('RISK_MANAGEMENT', False) and idx_dd252 != -1:
                if dd252_t < self.config['RISK_DRAWDOWN_TIER2']:
                    # Tier 2: Deep drawdown -> Force cash
                    final_action = 0.0
                    override_flag = "RISK_T2"
                    risk_override = True
                    override_counts['risk_max'] += 1
                elif dd252_t < self.config['RISK_DRAWDOWN_TIER1'] and sma_dist_t < 0:
                    # Tier 1: Warning zone + below SMA -> Reduce exposure
                    override_flag = "RISK_T1"
                    risk_override = True
                    position_scale = 0.5
                    override_counts['risk_mid'] += 1
            
            if not risk_override or override_flag == "RISK_T1":
                # ============================================================
                # PRIORITY 2: JUMP DETECTION OVERRIDE
                # ============================================================
                if l_stat_t > self.config['JUMP_THRESHOLD']:
                    # Bullish jump -> Force trend agent
                    final_action = self.agents['trend'].predict(obs_raw, deterministic=True)
                    override_flag = "JUMP+"
                    override_counts['jump_up'] += 1
                elif l_stat_t < -self.config['JUMP_THRESHOLD']:
                    # Bearish jump -> Force defensive agent
                    final_action = self.agents['defensive'].predict(obs_raw, deterministic=True)
                    override_flag = "JUMP-"
                    override_counts['jump_down'] += 1
                    
                # ============================================================
                # PRIORITY 3: MOMENTUM OVERRIDE
                # ============================================================
                elif mom_t > self.config['MOMENTUM_OVERRIDE_THRESHOLD']:
                    final_action = 1.0
                    override_flag = "MOM"
                    override_counts['momentum'] += 1
                    
                # ============================================================
                # PRIORITY 4: SOFT MOE ENSEMBLE
                # ============================================================
                else:
                    # Get regime probabilities
                    curr_feats = create_regime_features_df(ret_t, vol_t, vix_t, mom_t)
                    probs = self.regime_detector.predict_proba(curr_feats)[0]
                    weights = self.regime_detector.get_agent_weights(probs)
                    
                    # Detect conflict and get agent actions
                    conflict_detected, conflict_severity, agent_actions = self._detect_conflict(obs_raw)
                    
                    if conflict_detected:
                        # Reduce position when agents disagree, but not too much
                        position_scale *= (1 - self.config['CONFLICT_PENALTY'] * conflict_severity)
                        override_counts['conflict'] += 1
                    
                    # Weighted ensemble action
                    raw_action = sum(weights[k] * agent_actions[k] for k in agent_actions)
                    
                    # IMPORTANT: For LONG_ONLY mode, ensure adequate exposure when bullish consensus
                    bullish_count = sum(1 for a in agent_actions.values() if a > 0.1)
                    bearish_count = sum(1 for a in agent_actions.values() if a < -0.1)
                    
                    if self.config['LONG_ONLY']:
                        if bullish_count >= 3:
                            # Strong bullish consensus - ensure at least 50% exposure
                            raw_action = max(raw_action, 0.5)
                        elif bullish_count >= 2 and bearish_count == 0:
                            # Moderate bullish - ensure at least 30% exposure
                            raw_action = max(raw_action, 0.3)
                    
                    # Apply switching cost (minimal impact)
                    switching_penalty = self._calculate_switching_cost(weights)
                    
                    final_action = raw_action * (1 - switching_penalty)
                    
                    # Determine dominant regime for display
                    max_regime_idx = np.argmax(probs)
                    regime_name = self.regime_detector.get_regime_name(max_regime_idx)
                    override_flag = "MoE"
                    override_counts['moe'] += 1
            
            # ============================================================
            # EXECUTION
            # ============================================================
            mkt_ret = data_vals[t, idx_ret]
            
            # Vol targeting
            vol_scaler = 1.0
            if self.config['USE_VOL_TARGETING'] and vol_t > 0.01:
                vol_scaler = self.config['TARGET_VOL'] / vol_t
            vol_scaler = np.clip(vol_scaler, 0.1, 2.0)
            
            scaled_action = final_action * self.config['ACTION_SCALER'] * vol_scaler * position_scale
            
            if self.config['LONG_ONLY']:
                scaled_action = np.clip(scaled_action, 0, 10)
            
            # Apply leverage cap (with risk override)
            exposure_cap = self.config['MAX_LEVERAGE']
            if risk_override and override_flag == "RISK_T2":
                exposure_cap = 0.0
            
            position_size = np.clip(scaled_action, -exposure_cap, exposure_cap)
            
            # PnL calculation
            cost = abs(position_size - holdings) * self.config['FEES']
            lev_cost = max(0, abs(position_size) - 1) * self.config['BORROW_RATE']
            
            step_pnl_pct = (position_size * mkt_ret) - cost - lev_cost
            
            portfolio *= (1 + step_pnl_pct)
            benchmark_equity *= (1 + mkt_ret)
            holdings = position_size
            
            # Debug output
            if self.config['VERBOSE'] and t % self.config['PRINT_INTERVAL'] == 0:
                print(f"{str(dates[t].date()):<12} | {override_flag:<8} | {regime_name:<10} | {position_size:<8.2f} | {portfolio:>10,.0f} | {benchmark_equity:>10,.0f}")
            
            history.append({
                'Date': dates[t],
                'Portfolio': portfolio,
                'Return': step_pnl_pct,
                'Benchmark': mkt_ret,
                'Position': position_size,
                'Regime': regime_name,
                'Override': override_flag
            })
        
        # Results
        res = pd.DataFrame(history).set_index('Date')
        total_ret = (portfolio / self.config['INITIAL_BALANCE']) - 1
        bench_ret = (1 + res['Benchmark']).cumprod().iloc[-1] - 1
        alpha = total_ret - bench_ret
        
        print(f"\n{'=' * 60}")
        print("V13 MoE RESULTS")
        print(f"{'=' * 60}")
        print(f"Model Return:     {total_ret:>10.2%}")
        print(f"Benchmark Return: {bench_ret:>10.2%}")
        print(f"Alpha:            {alpha:>10.2%}")
        print(f"\nOverride Statistics:")
        for k, v in override_counts.items():
            if v > 0:
                print(f"  {k}: {v}")
        
        if plot_results:
            self._plot_dashboard(res)
        
        return total_ret
    
    def _plot_dashboard(self, res):
        """Generate comprehensive backtest visualization."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 2, 2, 2]})
        plt.subplots_adjust(hspace=0.3)
        
        # Calculate benchmark equity
        res['Bench_Equity'] = (1 + res['Benchmark']).cumprod() * self.config['INITIAL_BALANCE']
        
        # 1. Equity Curve
        ax0 = axes[0]
        ax0.plot(res.index, res['Portfolio'], label='V13 MoE', color='blue', linewidth=2)
        ax0.plot(res.index, res['Bench_Equity'], label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
        ax0.set_title(f"V13 MoE Equity Curve: {self.config['TARGET_ASSET']}", fontsize=14, fontweight='bold')
        ax0.set_ylabel("Portfolio Value ($)")
        ax0.legend(loc='upper left')
        ax0.grid(True, alpha=0.3)
        
        # Final stats annotation
        final_ret = (res['Portfolio'].iloc[-1] / self.config['INITIAL_BALANCE']) - 1
        bench_final = (res['Bench_Equity'].iloc[-1] / self.config['INITIAL_BALANCE']) - 1
        ax0.annotate(
            f"Model: {final_ret:.1%}\nBench: {bench_final:.1%}\nAlpha: {final_ret - bench_final:.1%}",
            xy=(0.02, 0.95), xycoords='axes fraction',
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # 2. Drawdown
        ax1 = axes[1]
        strat_peak = res['Portfolio'].cummax()
        strat_dd = (res['Portfolio'] - strat_peak) / strat_peak
        ax1.fill_between(res.index, strat_dd, 0, color='red', alpha=0.4, label='Strategy Drawdown')
        ax1.set_ylabel("Drawdown %")
        ax1.set_title("Drawdown", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 3. Position
        ax2 = axes[2]
        colors = ['forestgreen' if p > 0 else 'firebrick' for p in res['Position']]
        ax2.bar(res.index, res['Position'], color=colors, width=1.5)
        ax2.set_ylabel("Position Size")
        ax2.set_title("Position Over Time", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 4. Regime Distribution
        ax3 = axes[3]
        regime_colors = {
            'Growth': 'green',
            'Stagnation': 'gray',
            'Crisis': 'red',
            'Transition': 'orange',
            '-': 'blue'
        }
        for regime in res['Regime'].unique():
            mask = res['Regime'] == regime
            if mask.any():
                ax3.scatter(
                    res.index[mask],
                    [1] * mask.sum(),
                    c=regime_colors.get(regime, 'blue'),
                    label=regime,
                    alpha=0.7,
                    s=10
                )
        ax3.set_ylabel("Regime")
        ax3.set_title("Detected Regime Over Time", fontsize=12)
        ax3.legend(loc='upper right', ncol=4)
        ax3.set_ylim(0.5, 1.5)
        ax3.set_yticks([])
        
        plt.tight_layout()
        
        if self.config.get('SAVE_PLOTS', False):
            plot_path = os.path.join(
                self.config.get('PLOTS_DIR', './Plots'),
                f"v13_moe_{self.config['TARGET_ASSET']}.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {plot_path}")
        
        plt.close(fig)


if __name__ == "__main__":
    # Quick test with single ticker
    config = DEFAULT_CONFIG.copy()
    config['TICKERS'] = ["GOOG", "^VIX", "SHY"]
    config['TARGET_ASSET'] = "GOOG"
    config['LONG_ONLY'] = True
    config['TRAINING_STEPS'] = 30000  # Faster for testing
    config['PLOTS_DIR'] = "./Plots"
    
    print("\n>>> INITIALIZING V13 MoE MODEL <<<")
    mgr = MoEEnsembleManager(config)
    mgr.train_specialists()
    mgr.run_backtest()
