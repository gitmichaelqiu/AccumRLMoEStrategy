"""
Feature Engineering for Trading Strategy
Technical indicators and regime features for market analysis.
"""

import numpy as np
import pandas as pd


def calculate_lee_mykland(series, window=20):
    """
    Calculate Lee-Mykland Jump Detection statistic.
    
    The L-statistic measures standardized log returns against local volatility.
    Values > 3 indicate significant price jumps.
    
    Args:
        series: Price series
        window: Rolling window for volatility estimation
        
    Returns:
        Series of L-statistics
    """
    log_ret = np.log(series / series.shift(1))
    abs_ret = np.abs(log_ret)
    bv_terms = abs_ret * abs_ret.shift(1)
    local_vol = np.sqrt((np.pi / 2) * bv_terms.rolling(window=window).mean())
    l_stat = log_ret / (local_vol + 1e-9)
    return l_stat.fillna(0)


def calculate_adx(high, low, close, window=14):
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures trend strength (not direction).
    Values > 25 indicate strong trend, > 40 extreme trend.
    
    Args:
        high: High price series
        low: Low price series  
        close: Close price series
        window: Smoothing window
        
    Returns:
        Dictionary with 'adx', 'di_plus', 'di_minus' series
    """
    # Handle potential MultiIndex columns from yfinance
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    # Ensure we have Series
    high = pd.Series(high.values.flatten() if hasattr(high, 'values') else high, index=close.index)
    low = pd.Series(low.values.flatten() if hasattr(low, 'values') else low, index=close.index)
    close = pd.Series(close.values.flatten() if hasattr(close, 'values') else close, index=close.index)
    
    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        )
    )
    
    dm_plus = np.where(
        (high - high.shift(1)) > (low.shift(1) - low),
        np.maximum(high - high.shift(1), 0),
        0
    )
    dm_minus = np.where(
        (low.shift(1) - low) > (high - high.shift(1)),
        np.maximum(low.shift(1) - low, 0),
        0
    )
    
    # Convert to Series for rolling
    tr = pd.Series(tr, index=close.index)
    dm_plus = pd.Series(dm_plus, index=close.index)
    dm_minus = pd.Series(dm_minus, index=close.index)
    
    tr_s = tr.rolling(window).mean()
    dp_s = dm_plus.rolling(window).mean()
    dm_s = dm_minus.rolling(window).mean()
    
    di_plus = 100 * (dp_s / tr_s)
    di_minus = 100 * (dm_s / tr_s)
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
    adx = dx.rolling(window).mean()
    
    return {
        'adx': adx.fillna(0),
        'di_plus': di_plus.fillna(0),
        'di_minus': di_minus.fillna(0)
    }


def calculate_bollinger_bands(series, window=20, std_mult=2.0):
    """
    Calculate Bollinger Bands and derived features.
    
    Args:
        series: Price series
        window: SMA window
        std_mult: Standard deviation multiplier
        
    Returns:
        Dictionary with 'bb_width', 'pct_b' series
    """
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    bb_width = (std * std_mult * 2) / sma
    pct_b = (series - lower) / (upper - lower + 1e-9)
    
    return {
        'bb_width': bb_width.fillna(0),
        'pct_b': pct_b.fillna(0.5)
    }


def add_technical_features(df, ohlc, config):
    """
    Add all technical features to a dataframe.
    
    Args:
        df: DataFrame with price data (must have TARGET_ASSET column)
        ohlc: DataFrame with OHLC data
        config: Configuration dictionary with 'TARGET_ASSET', 'MOMENTUM_LOOKBACK'
        
    Returns:
        DataFrame with added features
    """
    target = config['TARGET_ASSET']
    if df.empty or target not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Basic returns
    df['returns'] = df[target].pct_change()
    
    # Jump Detection
    df['l_stat'] = calculate_lee_mykland(df[target], window=20)
    
    # ADX (Trend Strength)
    adx_data = calculate_adx(ohlc['High'], ohlc['Low'], ohlc['Close'])
    df['adx'] = adx_data['adx']
    df['di_plus'] = adx_data['di_plus']
    df['di_minus'] = adx_data['di_minus']
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(df[target])
    df['bb_width'] = bb_data['bb_width']
    df['pct_b'] = bb_data['pct_b']
    
    # VIX normalization (if available)
    if '^VIX' in df.columns:
        df['vix_norm'] = (df['^VIX'] - 15) / 40
    else:
        df['vix_norm'] = 0
    
    # SMA 200 distance
    sma200 = df[target].rolling(200).mean()
    df['dist_sma200'] = (df[target] - sma200) / sma200
    
    # Realized Volatility
    df['realized_vol_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)
    df['vol_percentile'] = df['realized_vol_20d'].rolling(252).rank(pct=True)
    
    # Momentum
    momentum_lookback = config.get('MOMENTUM_LOOKBACK', 50)
    df['momentum_50d'] = df[target].pct_change(momentum_lookback)
    
    # Drawdown
    rolling_high = df[target].rolling(50).max()
    df['drawdown'] = (df[target] - rolling_high) / rolling_high
    df['recovery_signal'] = (df['drawdown'] > -0.05) & (df['drawdown'].shift(10) < -0.10)
    
    # 252-day Drawdown for Risk Management
    rolling_high_252 = df[target].rolling(252).max()
    df['drawdown_252'] = (df[target] - rolling_high_252) / rolling_high_252
    
    return df.fillna(0)


def get_regime_features(df):
    """
    Extract features used for regime detection (GMM input).
    
    Args:
        df: DataFrame with technical features
        
    Returns:
        DataFrame with regime features: returns, realized_vol_20d, vix_norm, momentum_50d
    """
    regime_cols = ['returns', 'realized_vol_20d', 'vix_norm', 'momentum_50d']
    return df[regime_cols].fillna(0)
