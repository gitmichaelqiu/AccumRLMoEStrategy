#!/usr/bin/env python3
"""
Multi-Ticker Test for V13 MoE Model
Tests on American technology tickers with comprehensive benchmarking.
"""

import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import os

from v13_moe_model import DEFAULT_CONFIG, MoEEnsembleManager, DataProcessor

# Technology tickers for testing
TECH_TICKERS = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("GOOG", "Google"),
    ("NVDA", "NVIDIA"),
    ("TSLA", "Tesla"),
]

TEST_START = "2024-01-02"
TEST_END = "2025-02-05"


def run_single_ticker_test(ticker_symbol, ticker_name, verbose=True, max_retries=3):
    """Run V13 MoE backtest on a single ticker."""
    print(f"\n{'=' * 80}")
    print(f"V13 MoE TESTING: {ticker_name} ({ticker_symbol})")
    print(f"{'=' * 80}")
    
    for attempt in range(max_retries):
        try:
            config = DEFAULT_CONFIG.copy()
            config['TICKERS'] = [ticker_symbol, "^VIX", "SHY"]
            config['TARGET_ASSET'] = ticker_symbol
            config['TEST_START'] = TEST_START
            config['TEST_END'] = TEST_END
            
            # Test settings
            config['LONG_ONLY'] = True
            config['MAX_LEVERAGE'] = 1.0
            config['TRAINING_STEPS'] = 15000  # Balanced speed vs quality
            config['USE_VOL_TARGETING'] = True
            config['ACTION_SCALER'] = 5.0
            config['USE_SEED_ENSEMBLE'] = True
            config['SEEDS'] = [42, 123, 456]
            
            # V13 MoE specific
            config['SWITCHING_COST_LAMBDA'] = 0.02
            config['CONFLICT_THRESHOLD'] = 0.3
            config['CONFLICT_PENALTY'] = 0.3
            
            # Output
            config['SAVE_PLOTS'] = True
            config['PLOTS_DIR'] = './Plots'
            config['VERBOSE'] = verbose
            config['PRINT_INTERVAL'] = 30
            
            mgr = MoEEnsembleManager(config)
            mgr.train_specialists(verbose=verbose)
            
            total_return = mgr.run_backtest(plot_results=True)
            
            # Calculate benchmark return
            dp = DataProcessor([ticker_symbol], config)
            bench_data = dp.get_data(TEST_START, TEST_END)
            if not bench_data.empty and ticker_symbol in bench_data.columns:
                bench_returns = bench_data[ticker_symbol].pct_change().dropna()
                benchmark_return = (1 + bench_returns).prod() - 1
            else:
                benchmark_return = 0
            
            alpha = total_return - benchmark_return
            
            return {
                'ticker': ticker_symbol,
                'name': ticker_name,
                'model_return': total_return,
                'benchmark_return': benchmark_return,
                'alpha': alpha,
            }
            
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                traceback.print_exc()
    
    return None


def create_summary_plot(results, save_path="./Plots/v13_summary.png"):
    """Create a summary bar chart of all ticker results."""
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    tickers = [r['ticker'] for r in results]
    model_returns = [r['model_return'] * 100 for r in results]
    bench_returns = [r['benchmark_return'] * 100 for r in results]
    alphas = [r['alpha'] * 100 for r in results]
    
    x = np.arange(len(tickers))
    width = 0.25
    
    bars1 = ax.bar(x - width, model_returns, width, label='V13 MoE', color='blue', alpha=0.8)
    bars2 = ax.bar(x, bench_returns, width, label='Buy & Hold', color='gray', alpha=0.8)
    bars3 = ax.bar(x + width, alphas, width, label='Alpha', color='green', alpha=0.8)
    
    ax.set_xlabel('Ticker')
    ax.set_ylabel('Return (%)')
    ax.set_title('V13 MoE Performance Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSummary plot saved to: {save_path}")
    plt.close()


def run_comprehensive_test():
    """Run full test suite on all technology tickers."""
    print("\n" + "=" * 80)
    print("V13 MIXTURE OF EXPERTS MODEL EVALUATION")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print("=" * 80)
    print("\nKey V13 Features:")
    print("  - GMM-based Soft Gating (4 regimes)")
    print("  - Transaction Cost Penalty for smooth transitions")
    print("  - Conflict Detection (reduce position when agents disagree)")
    print("  - Priority Override: Risk > Jump > Momentum > MoE Ensemble")
    print("=" * 80)
    
    # Ensure Plots directory exists
    os.makedirs("./Plots", exist_ok=True)
    
    results = []
    
    for ticker, name in TECH_TICKERS:
        result = run_single_ticker_test(ticker, name, verbose=True)
        if result:
            results.append(result)
            print(f"\n>>> {name} ({ticker}): Model {result['model_return']:.2%} vs Benchmark {result['benchmark_return']:.2%}")
            print(f"    Alpha: {result['alpha']:.2%}")
        time.sleep(2)  # Rate limit buffer
    
    # Summary
    print("\n" + "=" * 80)
    print("V13 MoE COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    if results:
        summary_df = pd.DataFrame([{
            'Ticker': r['ticker'],
            'Name': r['name'],
            'Model Return': f"{r['model_return']:.2%}",
            'Benchmark': f"{r['benchmark_return']:.2%}",
            'Alpha': f"{r['alpha']:.2%}"
        } for r in results])
        
        print(summary_df.to_string(index=False))
        
        model_returns = [r['model_return'] for r in results]
        alphas = [r['alpha'] for r in results]
        
        print(f"\n{'=' * 60}")
        print("AGGREGATE STATISTICS")
        print(f"{'=' * 60}")
        print(f"Average Model Return:     {np.mean(model_returns):.2%}")
        print(f"Average Alpha:            {np.mean(alphas):.2%}")
        print(f"Win Rate (Alpha > 0):     {sum(1 for a in alphas if a > 0)}/{len(alphas)}")
        
        if len(alphas) > 0:
            best_idx = np.argmax(alphas)
            worst_idx = np.argmin(alphas)
            print(f"Best Performer:           {results[best_idx]['ticker']} ({alphas[best_idx]:.2%} alpha)")
            print(f"Worst Performer:          {results[worst_idx]['ticker']} ({alphas[worst_idx]:.2%} alpha)")
        
        # Create summary visualization
        create_summary_plot(results)
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_test()
