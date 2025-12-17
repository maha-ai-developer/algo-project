import os
import json
import infrastructure.config as config
from infrastructure.broker.kite_auth import get_kite

# Define location relative to the centralized CACHE_DIR
INSTRUMENTS_FILE = os.path.join(config.CACHE_DIR, "instruments.json")

def refresh_instrument_cache():
    """
    Downloads full instrument list from Kite and saves to JSON.
    This creates a fast local lookup file.
    """
    print("🔄 Downloading Instrument Master Dump...")
    try:
        kite = get_kite()
        instruments = kite.instruments() # Returns list of dicts (~6MB)
        
        # Convert to dictionary for fast lookup: {'NSE:INFY': 408065}
        lookup = {}
        for instr in instruments:
            key = f"{instr['exchange']}:{instr['tradingsymbol']}"
            lookup[key] = instr['instrument_token']
            
            # Also save raw symbol for default exchange lookups (NSE preference)
            if instr['exchange'] == 'NSE':
                lookup[instr['tradingsymbol']] = instr['instrument_token']

        # Save to disk
        with open(INSTRUMENTS_FILE, "w") as f:
            json.dump(lookup, f)
            
        print(f"✅ Cached {len(lookup)} instruments to {INSTRUMENTS_FILE}")
        
    except Exception as e:
        print(f"❌ Failed to refresh instruments: {e}")

def get_instrument_token(symbol, exchange="NSE"):
    """
    Returns instrument_token for a symbol (e.g., 'INFY' -> 408065).
    Auto-refreshes cache if missing.
    """
    if not os.path.exists(INSTRUMENTS_FILE):
        refresh_instrument_cache()
        
    try:
        with open(INSTRUMENTS_FILE, "r") as f:
            lookup = json.load(f)
            
        # 1. Try exact match "NSE:INFY"
        key = f"{exchange}:{symbol}"
        if key in lookup:
            return lookup[key]
            
        # 2. Try symbol only "INFY" (Defaults to NSE usually)
        if symbol in lookup:
            return lookup[symbol]
            
        # 3. If not found, force refresh and try once more
        print(f"⚠️ Token for {symbol} not found. Refreshing cache...")
        refresh_instrument_cache()
        
        with open(INSTRUMENTS_FILE, "r") as f:
            lookup = json.load(f)
            
        if key in lookup: return lookup[key]
        if symbol in lookup: return lookup[symbol]
        
    except Exception as e:
        print(f"❌ Token Lookup Error: {e}")
        
    return None
