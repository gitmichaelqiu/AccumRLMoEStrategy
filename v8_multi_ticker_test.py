#!/usr/bin/env python3
"""
Multi-Ticker Test for V8 Trading Model (Improved)
Compares V8 performance across tech tickers
"""

import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Suppress popups
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Import from v8Model
from v8Model import (
    DEFAULT_CONFIG, 
    DataProcessor, 
    EnsembleManager, 
    HyperparameterOptimizer,
    run_system
)

# Tech tickers to test
TECH_TICKERS = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("GOOG", "Google"),
    ("NVDA", "NVIDIA"),
    ("TSLA", "Tesla"),
]

# Test period
TEST_START = "2024-01-02"
TEST_END = "2025-02-05"

def run_single_ticker_test(ticker_symbol, ticker_name, optimize=False, verbose=True, max_retries=3):
    """Run V8 model on a single ticker and return performance metrics"""
    
    print(f"\n{'='*80}")
    print(f"V8 TESTING: {ticker_name} ({ticker_symbol})")
    print(f"{'='*80}")
    
    for attempt in range(max_retries):
        try:
            config = DEFAULT_CONFIG.copy()
            config['TICKERS'] = [ticker_symbol, "^VIX", "SHY"]
            config['TARGET_ASSET'] = ticker_symbol
            config['TEST_START'] = TEST_START
            config['TEST_END'] = TEST_END
            
            # V8 improved settings for single stocks
            config['LONG_ONLY'] = True
            config['MAX_LEVERAGE'] = 1.0
            config['TRAINING_STEPS'] = 10000  # Fast for testing
            config['USE_VOL_TARGETING'] = False
            config['ACTION_SCALER'] = 10.0
            
            # V8 NEW: Less conservative bear settings
            config['BEAR_TREND_MIN_POS'] = 0.25
            config['BEAR_TREND_MAX_POS'] = 0.75
            config['ADX_THRESHOLD'] = 30  # Higher to reduce false bear signals
            config['MOMENTUM_THRESHOLD'] = 0.15
            config['MOMENTUM_LOOKBACK'] = 50
            
            # Save plots
            config['SAVE_PLOTS'] = True
            config['PLOTS_DIR'] = './plots'
            
            if optimize:
                opt = HyperparameterOptimizer(config['TICKERS'], ticker_symbol)
                best_params = opt.optimize("2015-01-01", "2023-12-31")
                config.update(best_params)
            
            mgr = EnsembleManager(config)
            mgr.train_specialists(verbose=verbose)
            
            # Run backtest and capture results
            warmup_dt = pd.Timestamp(TEST_START) - pd.Timedelta(days=365)
            full_data = mgr.dp.get_data(warmup_dt.strftime('%Y-%m-%d'), TEST_END)
            
            if full_data.empty:
                print(f"  Empty data for {ticker_symbol}, retrying...")
                time.sleep(2)
                continue
                
            test_indices = np.where((full_data.index >= TEST_START) & (full_data.index <= TEST_END))[0]
            if len(test_indices) == 0:
                print(f"  No test indices for {ticker_symbol}, retrying...")
                time.sleep(2)
                continue
            
            # Run the actual backtest
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
                'config': config
            }
            
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed for {ticker_symbol}: {e}")
            if attempt < max_retries - 1:
                print(f"  Waiting 5 seconds before retry...")
                time.sleep(5)
            else:
                print(f"  All retries exhausted for {ticker_symbol}")
                traceback.print_exc()
    
    return None


def run_comprehensive_test(optimize=False):
    """Run V8 tests across all tech tickers and summarize results"""
    
    print("\n" + "="*80)
    print("V8 MODEL COMPREHENSIVE MULTI-TICKER EVALUATION")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print("Key V8 Improvements:")
    print("  - ADX threshold: 30 (was 25)")
    print("  - Momentum override at 15% 50-day return")
    print("  - Minimum 25% position in BEAR_TREND (was 0%)")
    print("  - 10% crisis floor (was 0%)")
    print("="*80)
    
    results = []
    
    for ticker, name in TECH_TICKERS:
        result = run_single_ticker_test(ticker, name, optimize=optimize, verbose=True)
        if result:
            results.append(result)
            print(f"\n>>> {name} ({ticker}): Model {result['model_return']:.2%} vs Benchmark {result['benchmark_return']:.2%}")
            print(f"    Alpha: {result['alpha']:.2%}")
        
        # Brief pause between tickers to avoid rate limiting
        time.sleep(2)
    
    # Summary
    print("\n" + "="*80)
    print("V8 COMPREHENSIVE SUMMARY")
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
        
        # Aggregate stats
        model_returns = [r['model_return'] for r in results]
        bench_returns = [r['benchmark_return'] for r in results]
        alphas = [r['alpha'] for r in results]
        
        print(f"\n{'='*60}")
        print("V8 AGGREGATE STATISTICS")
        print(f"{'='*60}")
        print(f"Average Model Return:     {np.mean(model_returns):.2%}")
        print(f"Average Benchmark Return: {np.mean(bench_returns):.2%}")
        print(f"Average Alpha:            {np.mean(alphas):.2%}")
        print(f"Win Rate (Alpha > 0):     {sum(1 for a in alphas if a > 0)}/{len(alphas)}")
        
        if len(alphas) > 0:
            print(f"Best Performer:           {results[np.argmax(alphas)]['ticker']} ({max(alphas):.2%} alpha)")
            print(f"Worst Performer:          {results[np.argmin(alphas)]['ticker']} ({min(alphas):.2%} alpha)")
    else:
        print("No successful tests completed due to data issues.")
    
    return results


if __name__ == "__main__":
    # Run without optimization for faster initial test
    results = run_comprehensive_test(optimize=False)
