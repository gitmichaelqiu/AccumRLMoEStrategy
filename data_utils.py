"""
Data Utilities for Trading Strategy
Provides data fetching with anti-rate-limit session using curl_cffi.
"""

import yfinance as yf
import pandas as pd
from curl_cffi import requests


def create_session():
    """Create a Chrome-impersonated session to bypass rate limits."""
    return requests.Session(impersonate="chrome")


def download_data(tickers, start, end, session=None):
    """
    Download Close price data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        session: Optional curl_cffi session
        
    Returns:
        DataFrame with Close prices for each ticker
    """
    if session is None:
        session = create_session()
    
    try:
        print(f"Fetching data for: {tickers} ({start} to {end})")
        data = yf.download(tickers, start=start, end=end, progress=False, session=session)
        
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                data = data.xs('Close', level=0, axis=1)
            elif 'Adj Close' in data.columns.levels[0]:
                data = data.xs('Adj Close', level=0, axis=1)
            elif 'Close' in data.columns.levels[1]:
                data = data.xs('Close', level=1, axis=1)
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = [tickers[0]] if isinstance(tickers, list) else [tickers]
        
        return data
    except Exception as e:
        print(f"Data Download Error: {e}")
        return pd.DataFrame()


def download_ohlc(ticker, start, end, session=None):
    """
    Download OHLC data for a single ticker.
    
    Args:
        ticker: Single ticker symbol
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        session: Optional curl_cffi session
        
    Returns:
        DataFrame with OHLC columns
    """
    if session is None:
        session = create_session()
    
    try:
        ohlc = yf.download(ticker, start=start, end=end, progress=False, session=session)
        return ohlc
    except Exception as e:
        print(f"OHLC Download Error: {e}")
        return pd.DataFrame()


def download_with_ohlc(tickers, target_asset, start, end, session=None):
    """
    Download both Close prices and OHLC for target asset.
    
    Args:
        tickers: List of ticker symbols
        target_asset: The main asset to get OHLC for
        start: Start date string
        end: End date string
        session: Optional curl_cffi session
        
    Returns:
        Tuple of (close_prices_df, ohlc_df)
    """
    if session is None:
        session = create_session()
    
    # Ensure target asset is in tickers
    if target_asset not in tickers:
        tickers = list(tickers) + [target_asset]
    
    close_data = download_data(tickers, start, end, session)
    ohlc_data = download_ohlc(target_asset, start, end, session)
    
    return close_data, ohlc_data
