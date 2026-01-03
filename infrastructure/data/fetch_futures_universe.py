"""
Fetch Futures Universe from Kite API

Dynamically fetches all active Stock Futures from NFO segment,
extracts unique underlying symbols, and saves to data/universe/futures_symbols.txt.

Usage:
    python cli.py fetch_universe
    
    Or directly:
    python infrastructure/data/fetch_futures_universe.py
"""

import os
import sys

# Ensure root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import infrastructure.config as config
from infrastructure.broker.kite_auth import get_kite


# Indices to exclude (we only want stock futures, not index futures)
INDEX_NAMES = {
    'NIFTY', 
    'BANKNIFTY', 
    'FINNIFTY', 
    'NIFTYNXT50', 
    'MIDCPNIFTY',
    'SENSEX',
    'BANKEX',
}


def fetch_futures_universe(save_path: str = None) -> list:
    """
    Fetch all active Stock Futures symbols from Kite NFO segment.
    
    Args:
        save_path: Optional path to save symbols. Defaults to data/universe/futures_symbols.txt
        
    Returns:
        List of unique stock symbols that have active futures contracts
    """
    print("\n--- 📊 FETCHING FUTURES UNIVERSE FROM KITE ---\n")
    
    # 1. Connect to Kite
    print("   🔑 Connecting to Kite...")
    try:
        kite = get_kite()
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        print("   💡 Run 'python cli.py login' first to authenticate")
        return []
    
    # 2. Fetch NFO Instruments
    print("   📥 Fetching NFO instruments...")
    try:
        instruments = kite.instruments("NFO")
        print(f"   ✅ Loaded {len(instruments):,} NFO instruments")
    except Exception as e:
        print(f"   ❌ Failed to fetch instruments: {e}")
        return []
    
    # 3. Filter for Futures only
    futures = [i for i in instruments if i.get('instrument_type') == 'FUT']
    print(f"   📋 Found {len(futures):,} Futures contracts")
    
    # 4. Exclude Index Futures
    stock_futures = [f for f in futures if f.get('name', '').upper() not in INDEX_NAMES]
    print(f"   🏢 Stock Futures (excluding indices): {len(stock_futures):,}")
    
    # 5. Extract Unique Underlying Names
    unique_symbols = sorted(set(f['name'] for f in stock_futures if f.get('name')))
    print(f"   ✨ Unique underlying stocks: {len(unique_symbols)}")
    
    # 6. Save to file
    if save_path is None:
        save_path = os.path.join(config.UNIVERSE_DIR, "futures_symbols.txt")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        for symbol in unique_symbols:
            f.write(f"{symbol}\n")
    
    print(f"\n   💾 Saved to: {save_path}")
    print(f"   📊 Total symbols: {len(unique_symbols)}")
    
    # Show sample
    print(f"\n   📝 Sample (first 10):")
    for sym in unique_symbols[:10]:
        print(f"      • {sym}")
    if len(unique_symbols) > 10:
        print(f"      ... and {len(unique_symbols) - 10} more")
    
    print("\n   ✅ Done!\n")
    
    return unique_symbols


def load_futures_universe(file_path: str = None) -> list:
    """
    Load futures universe from saved file.
    
    Args:
        file_path: Path to symbols file. Defaults to data/universe/futures_symbols.txt
        
    Returns:
        List of stock symbols
    """
    if file_path is None:
        file_path = os.path.join(config.UNIVERSE_DIR, "futures_symbols.txt")
    
    if not os.path.exists(file_path):
        print(f"   ⚠️ Universe file not found: {file_path}")
        print("   💡 Run 'python cli.py fetch_universe' first")
        return []
    
    with open(file_path, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    return symbols


if __name__ == "__main__":
    # Direct execution for testing
    fetch_futures_universe()
