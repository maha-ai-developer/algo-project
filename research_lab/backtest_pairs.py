import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.data_manager import download_historical_data

def run_backtest():
    print("--- 🔙 PAIRS BACKTEST (OUT-OF-SAMPLE & COST ADJUSTED) ---")
    
    # 💰 REALITY CHECK PARAMETERS
    CAPITAL_PER_LEG = 5000   # Fixed Capital per leg
    COST_PCT = 0.001         # 0.1% per trade (Entry + Exit)
    ENTRY_Z = 2.5            # Strict Entry
    EXIT_Z = 0.0             # Mean Reversion
    
    # 1. Load Candidates
    if not os.path.exists(config.PAIRS_CANDIDATES_FILE):
        print("❌ Run 'scan_pairs' first.")
        return

    with open(config.PAIRS_CANDIDATES_FILE, "r") as f:
        candidates = json.load(f)

    # 2. Download TEST Data (Last 60 Days)
    print("\n⬇️ Fetching Test Data (60 Days 5-min)...")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    unique_symbols = set()
    for p in candidates:
        unique_symbols.add(p['leg1'])
        unique_symbols.add(p['leg2'])
        
    download_historical_data(list(unique_symbols), from_date, to_date, interval="5minute")

    results = []

    print("\n🧪 Running Simulation...")
    for pair in candidates:
        s1, s2 = pair['leg1'], pair['leg2']
        
        try:
            # Load Data
            df1 = pd.read_csv(os.path.join(config.HISTORICAL_DIR, f"{s1}_5m.csv"))
            df2 = pd.read_csv(os.path.join(config.HISTORICAL_DIR, f"{s2}_5m.csv"))
            
            # Align Indices
            df1['date'] = pd.to_datetime(df1['date'])
            df2['date'] = pd.to_datetime(df2['date'])
            df1.set_index('date', inplace=True)
            df2.set_index('date', inplace=True)
            
            # Inner Join
            df = pd.merge(df1['close'], df2['close'], left_index=True, right_index=True, suffixes=('_1', '_2')).dropna()
        except Exception as e:
            continue

        if df.empty: continue

        # Logic
        ratio = df['close_1'] / df['close_2']
        zscore = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std()
        
        position = 0 
        pnl_cash = 0.0
        trades = 0
        entry_p1, entry_p2 = 0, 0
        
        # Simulation Loop
        for i in range(20, len(df)):
            z = zscore.iloc[i]
            p1 = df['close_1'].iloc[i]
            p2 = df['close_2'].iloc[i]
            
            # ENTRY
            if position == 0:
                if z > ENTRY_Z: # SHORT SPREAD
                    position = -1
                    entry_p1, entry_p2 = p1, p2
                    pnl_cash -= (CAPITAL_PER_LEG * 2) * COST_PCT 
                    trades += 1
                    
                elif z < -ENTRY_Z: # LONG SPREAD
                    position = 1
                    entry_p1, entry_p2 = p1, p2
                    pnl_cash -= (CAPITAL_PER_LEG * 2) * COST_PCT
                    trades += 1

            # EXIT
            elif position != 0:
                if abs(z) < EXIT_Z: # MEAN REVERSION
                    # Dollar Neutral Sizing
                    qty1 = int(CAPITAL_PER_LEG / entry_p1)
                    qty2 = int(CAPITAL_PER_LEG / entry_p2)
                    
                    if position == -1: 
                        diff1 = (entry_p1 - p1) * qty1 
                        diff2 = (p2 - entry_p2) * qty2 
                    else: 
                        diff1 = (p1 - entry_p1) * qty1 
                        diff2 = (entry_p2 - p2) * qty2 
                        
                    pnl_cash += (diff1 + diff2)
                    pnl_cash -= (CAPITAL_PER_LEG * 2) * COST_PCT 
                    position = 0

        if trades > 0:
            results.append({
                "Pair": f"{s1}-{s2}",
                "Sector": pair.get('sector', 'N/A'),
                "Trades": trades,
                "Net PnL (₹)": round(pnl_cash, 2)
            })

    # Display Top Results
    if results:
        df_res = pd.DataFrame(results).sort_values(by="Net PnL (₹)", ascending=False)
        print("\n" + tabulate(df_res.head(10), headers="keys", tablefmt="simple_grid"))
        
        # Auto-Select Champion
        best = df_res.iloc[0]
        s1, s2 = best['Pair'].split('-')
        print(f"\n🏆 Champion Pair: {best['Pair']} (Profit: ₹{best['Net PnL (₹)']})")
        
        cfg = [{
            "leg1": s1, "leg2": s2,
            "entry_z": ENTRY_Z, "exit_z": EXIT_Z
        }]
        with open(config.PAIRS_CONFIG, "w") as f:
            json.dump(cfg, f, indent=4)
        print("✅ Updated pairs_config.json with Champion Pair.")
    else:
        print("❌ No profitable pairs found in backtest.")

if __name__ == "__main__":
    run_backtest()
