#!/usr/bin/env python3
"""
Multi-Ticker Test for V9 Trading Model
Tests seed ensemble + 4-agent weighted voting
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

from v9Model import (
    DEFAULT_CONFIG, 
    DataProcessor, 
    EnsembleManager,
    set_seed
)

# Tech tickers to test
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
    """Run V9 model on a single ticker"""
    
    print(f"\n{'='*80}")
    print(f"V9 TESTING: {ticker_name} ({ticker_symbol})")
    print(f"{'='*80}")
    
    for attempt in range(max_retries):
        try:
            config = DEFAULT_CONFIG.copy()
            config['TICKERS'] = [ticker_symbol, "^VIX", "SHY"]
            config['TARGET_ASSET'] = ticker_symbol
            config['TEST_START'] = TEST_START
            config['TEST_END'] = TEST_END
            
            # V9 settings
            config['LONG_ONLY'] = True
            config['MAX_LEVERAGE'] = 1.0
            config['TRAINING_STEPS'] = 10000  # Faster for testing
            config['USE_VOL_TARGETING'] = False
            config['ACTION_SCALER'] = 10.0
            config['USE_SEED_ENSEMBLE'] = True
            config['USE_WEIGHTED_ENSEMBLE'] = True
            config['SEEDS'] = [42, 123, 456]
            
            config['SAVE_PLOTS'] = True
            config['PLOTS_DIR'] = './plots'
            
            mgr = EnsembleManager(config)
            mgr.train_specialists(verbose=verbose)
            
            # Run backtest
            total_return = mgr.run_backtest(plot_results=verbose)
            
            # Calculate benchmark return
            dp = DataProcessor([ticker_symbol], config)
            data, _ = dp.download(TEST_START, TEST_END)
            if not data.empty and ticker_symbol in data.columns:
                bench_returns = data[ticker_symbol].pct_change().dropna()
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
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Waiting 5 seconds...")
                time.sleep(5)
            else:
                traceback.print_exc()
    
    return None


def run_comprehensive_test():
    """Run V9 tests across all tech tickers"""
    
    print("\n" + "="*80)
    print("V9 MODEL COMPREHENSIVE EVALUATION")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print("Key V9 Features:")
    print("  - Seed ensemble: 3 seeds [42, 123, 456]")
    print("  - 4 agents: Trend, MeanRev, Defensive, Recovery")
    print("  - Weighted voting: Dynamic weights based on indicators")
    print("="*80)
    
    results = []
    
    for ticker, name in TECH_TICKERS:
        result = run_single_ticker_test(ticker, name, verbose=True)
        if result:
            results.append(result)
            print(f"\n>>> {name} ({ticker}): Model {result['model_return']:.2%} vs Benchmark {result['benchmark_return']:.2%}")
            print(f"    Alpha: {result['alpha']:.2%}")
        time.sleep(2)
    
    # Summary
    print("\n" + "="*80)
    print("V9 COMPREHENSIVE SUMMARY")
    print("="*80)
    
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
        bench_returns = [r['benchmark_return'] for r in results]
        alphas = [r['alpha'] for r in results]
        
        print(f"\n{'='*60}")
        print("V9 AGGREGATE STATISTICS")
        print(f"{'='*60}")
        print(f"Average Model Return:     {np.mean(model_returns):.2%}")
        print(f"Average Benchmark Return: {np.mean(bench_returns):.2%}")
        print(f"Average Alpha:            {np.mean(alphas):.2%}")
        print(f"Win Rate (Alpha > 0):     {sum(1 for a in alphas if a > 0)}/{len(alphas)}")
        
        if len(alphas) > 0:
            print(f"Best Performer:           {results[np.argmax(alphas)]['ticker']} ({max(alphas):.2%} alpha)")
            print(f"Worst Performer:          {results[np.argmin(alphas)]['ticker']} ({min(alphas):.2%} alpha)")
    else:
        print("No successful tests completed.")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_test()
