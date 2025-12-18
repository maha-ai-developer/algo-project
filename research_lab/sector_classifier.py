import pandas as pd
import os
import sys
import json

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infrastructure.config as config
from infrastructure.llm.client import GeminiAgent # <--- Importing the CLASS now

def load_universe_symbols():
    """Reads the Nifty 50 CSV from data/universe/"""
    universe_dir = config.UNIVERSE_DIR
    
    # Find the CSV file
    files = [f for f in os.listdir(universe_dir) if f.endswith(".csv")]
    if not files:
        print(f"❌ No Universe CSV found in {universe_dir}")
        return []
    
    csv_path = os.path.join(universe_dir, files[0])
    print(f"📂 Loading Universe from: {files[0]}")
    
    try:
        df = pd.read_csv(csv_path)
        # Clean column names
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Look for SYMBOL column
        if 'SYMBOL' in df.columns:
            return df['SYMBOL'].tolist()
        else:
            # Fallback: Use first column
            return df.iloc[:, 0].tolist()
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return []

def get_sector_map():
    """
    Orchestrator: Check Cache -> Load Symbols -> Ask AI -> Save Cache
    """
    # 1. Check if cache exists
    if os.path.exists(config.SECTOR_CACHE_FILE):
        print(f"⚡ Loading Sectors from Cache: {config.SECTOR_CACHE_FILE}")
        with open(config.SECTOR_CACHE_FILE, "r") as f:
            return json.load(f)

    # 2. Load Symbols
    symbols = load_universe_symbols()
    if not symbols: return {}

    # 3. Initialize AI Agent
    try:
        agent = GeminiAgent()  # <--- Creating the Agent
        sector_map = agent.classify_sectors(symbols) # <--- Calling the method
    except Exception as e:
        print(f"❌ Agent Init Failed: {e}")
        return {}
    
    # 4. Save to Cache
    if sector_map:
        with open(config.SECTOR_CACHE_FILE, "w") as f:
            json.dump(sector_map, f, indent=4)
        print(f"✅ Sector Map Saved to: {config.SECTOR_CACHE_FILE}")
    
    return sector_map

if __name__ == "__main__":
    get_sector_map()
