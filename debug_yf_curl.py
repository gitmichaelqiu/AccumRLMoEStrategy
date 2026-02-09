
from curl_cffi import requests
import yfinance as yf
import pandas as pd

# Create a session with chrome impersonation
session = requests.Session(impersonate="chrome")

print("Testing yfinance with curl_cffi session...")

try:
    # Method 1: Ticker object
    print("\n--- Method 1: yf.Ticker ---")
    goog = yf.Ticker("GOOG", session=session)
    hist = goog.history(period="1mo")
    print("Ticker history head:\n", hist.head())
    
    if not hist.empty:
        print("Success via Ticker object!")
    else:
        print("Failed via Ticker object (Empty)")

    # Method 2: yf.download (if supported)
    # yfinance.download doesn't explicitly take 'session' in all versions, 
    # but let's see if we can pass it or if it uses the global/thread-local one if we could set it.
    # Actually, yfinance 0.2.x might not support session in download() directly without overrides.
    # But let's try just in case.
    print("\n--- Method 2: yf.download ---")
    try:
        data = yf.download(["GOOG", "^VIX"], period="1mo", session=session)
        print("Download head:\n", data.head())
    except TypeError as te:
        print(f"yf.download rejected session arg: {te}")
    except Exception as e:
        print(f"yf.download failed: {e}")

except Exception as e:
    print(f"Global Fail: {e}")
    import traceback
    traceback.print_exc()
