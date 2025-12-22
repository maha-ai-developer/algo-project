import pandas as pd
import os
import sys
import json
import itertools
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.data_manager import download_historical_data
from strategies.stat_arb_bot import StatArbBot

def scan_pairs():
    print("--- 🔬 STAT ARB SCANNER (METHOD 2: COINTEGRATION) ---")

    # 1. Load Sector Universe (Leaders & Challengers Only)
    if not os.path.exists(config.SECTOR_REPORT_FILE):
        print("❌ Sector Report not found. Run 'sector_analysis' first.")
        return

    df_sector = pd.read_csv(config.SECTOR_REPORT_FILE)
    df_valid = df_sector[df_sector['Position'].isin(['LEADER', 'CHALLENGER'])].copy()
    
    sector_groups = df_valid.groupby('Broad_Sector')['Symbol'].apply(list).to_dict()
    all_symbols = df_valid['Symbol'].unique().tolist()
    
    print(f"💎 Universe: {len(all_symbols)} Stocks.")
    
    # 2. Bulk Download (1 Year Data)
    print("\n⬇️ Fetching Data...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    download_historical_data(all_symbols, start_date, end_date, interval="day")

    # 3. Load Data to RAM
    price_cache = {}
    for symbol in all_symbols:
        path = os.path.join(config.DATA_DIR, f"{symbol}_day.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            price_cache[symbol] = df['close']

    # 4. Scan Loop
    bot = StatArbBot() # Initialize Brain
    candidates = []
    
    total_pairs = sum([(len(s)*(len(s)-1))//2 for s in sector_groups.values() if len(s)>=2])
    count = 0
    
    print(f"\n⚙️ Running OLS Regression & ADF Tests on {total_pairs} pairs...")

    for sector, symbols in sector_groups.items():
        if len(symbols) < 2: continue
        
        for s1, s2 in list(itertools.combinations(symbols, 2)):
            count += 1
            sys.stdout.write(f"\r   👉 [{count}/{total_pairs}] {sector}: {s1} vs {s2}...")
            sys.stdout.flush()

            if s1 not in price_cache or s2 not in price_cache: continue
            
            try:
                # Use Bot to Check Cointegration
                is_valid = bot.calibrate(price_cache[s1], price_cache[s2], s1, s2)
                
                if is_valid:
                    # Logic: Method 2 passed (Stationary Residuals)
                    print(f"\n      ✅ FOUND: {bot.y_symbol} (Y) vs {bot.x_symbol} (X) | Beta: {bot.beta:.3f}")
                    
                    candidates.append({
                        "leg1": bot.y_symbol, # Dependent
                        "leg2": bot.x_symbol, # Independent
                        "sector": sector,
                        "hedge_ratio": round(bot.beta, 4), # CRITICAL: Save Beta
                        "intercept": round(bot.intercept, 4)
                    })
                    
            except Exception:
                continue

    # 5. Save
    print("\n")
    if candidates:
        with open(config.PAIRS_CANDIDATES_FILE, "w") as f:
            json.dump(candidates, f, indent=4)
        print(f"✅ Saved {len(candidates)} Cointegrated Pairs to {config.PAIRS_CANDIDATES_FILE}")
    else:
        print("❌ No cointegrated pairs found.")

if __name__ == "__main__":
    scan_pairs()
