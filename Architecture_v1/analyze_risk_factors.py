import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_risk_factors(ticker="SPY", start="2015-01-01", end="2023-12-31"):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.levels[0]: df = df.xs('Close', level=0, axis=1)
        elif 'Adj Close' in df.columns.levels[0]: df = df.xs('Adj Close', level=0, axis=1)
    
    if isinstance(df, pd.Series): df = df.to_frame()
    if ticker not in df.columns: df.columns = [ticker]
    
    # Calculate Indicators
    close = df[ticker]
    df['Returns'] = close.pct_change()
    df['Next_Return'] = df['Returns'].shift(-1) # Target variable
    
    # 1. Moving Averages
    df['SMA50'] = close.rolling(50).mean()
    df['SMA200'] = close.rolling(200).mean()
    df['Dist_SMA50'] = (close - df['SMA50']) / df['SMA50']
    df['Dist_SMA200'] = (close - df['SMA200']) / df['SMA200']
    
    # 2. Momentum
    df['Mom_20d'] = close.pct_change(20)
    df['Mom_50d'] = close.pct_change(50)
    
    # 3. Drawdown
    rolling_max = close.rolling(252, min_periods=1).max()
    df['Drawdown'] = (close - rolling_max) / rolling_max
    
    # 4. Volatility
    df['Vol_20d'] = df['Returns'].rolling(20).std() * np.sqrt(252)
    
    # Drop NaNs
    df = df.dropna()
    
    # Analysis: Negative Returns Conditional Probability
    print("\n--- RISK FACTOR ANALYSIS --- (Predicting Negative Next Day Return)")
    
    # Define thresholds to test
    conditions = [
        ('Price < SMA50', df['Dist_SMA50'] < 0),
        ('Price < SMA200', df['Dist_SMA200'] < 0),
        ('Price < SMA200 - 5%', df['Dist_SMA200'] < -0.05),
        ('Mom_50d < 0', df['Mom_50d'] < 0),
        ('Mom_50d < -0.05', df['Mom_50d'] < -0.05),
        ('Drawdown < -10%', df['Drawdown'] < -0.10),
        ('Drawdown < -20%', df['Drawdown'] < -0.20),
        ('High Vol (>20%)', df['Vol_20d'] > 0.20),
        ('Bear Combo (Price<SMA200 & Mom50<0)', (df['Dist_SMA200'] < 0) & (df['Mom_50d'] < 0))
    ]
    
    stats = []
    
    global_neg_prob = (df['Next_Return'] < 0).mean()
    global_avg_ret = df['Next_Return'].mean() * 100
    print(f"Global Probability of Negative Return: {global_neg_prob:.2%}")
    print(f"Global Average Daily Return: {global_avg_ret:.4f}%")
    print("-" * 80)
    print(f"{'Condition':<35} | {'Count':<6} | {'Neg Prob (Risk)':<15} | {'Avg Next Ret':<15}")
    print("-" * 80)
    
    for name, condition in conditions:
        subset = df[condition]
        if len(subset) > 0:
            prob = (subset['Next_Return'] < 0).mean()
            avg = subset['Next_Return'].mean() * 100
            stats.append({
                'Condition': name,
                'Count': len(subset),
                'Neg_Prob': prob,
                'Avg_Ret': avg
            })
            print(f"{name:<35} | {len(subset):<6} | {prob:<15.2%} | {avg:<15.4f}%")
    
    print("-" * 80)
    
    # Suggestion
    best_risk = max(stats, key=lambda x: x['Neg_Prob'])
    print(f"\n>> HIGHEST RISK SIGNAL: {best_risk['Condition']} (Neg Prob: {best_risk['Neg_Prob']:.2%})")
    
    worst_return = min(stats, key=lambda x: x['Avg_Ret'])
    print(f">> WORST RETURN SIGNAL: {worst_return['Condition']} (Avg Ret: {worst_return['Avg_Ret']:.4f}%)")

if __name__ == "__main__":
    analyze_risk_factors("SPY")
    analyze_risk_factors("GOOG") # Check single stock behavior too
