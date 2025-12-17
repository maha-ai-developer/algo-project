import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
from tabulate import tabulate

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.data_manager import download_historical_data, DataManager

def calculate_zscore(series_a, series_b, window=20):
    ratio = series_a / series_b
    r_mean = ratio.rolling(window=window).mean()
    r_std = ratio.rolling(window=window).std()
    return (ratio - r_mean) / r_std

def run_backtest():
    print("--- 🔙 PHASE 3b: PAIRS BACKTEST & SELECTION ---")
    
    # 1. Load Candidates
    if not os.path.exists(config.PAIRS_CANDIDATES_FILE):
        print("❌ Candidates missing. Run scan_pairs.py first.")
        return

    with open(config.PAIRS_CANDIDATES_FILE, "r") as f:
        candidates = json.load(f)

    # 2. Download 5-min Data (Batch)
    pairs_txt_path = os.path.join(config.ARTIFACTS_DIR, "pairs_dataset.txt")
    if os.path.exists(pairs_txt_path):
        with open(pairs_txt_path, "r") as f:
            symbols = [s.strip() for s in f.readlines() if s.strip()]
            
        print(f"⬇️ Fetching 1 Year 5-min Data for {len(symbols)} symbols...")
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        download_historical_data(symbols, from_date, to_date, interval="5m")

    # 3. Run Simulation
    results = []
    
    print("\n🏎️  Running Simulation on Candidates...")
    for pair in candidates:
        s1, s2 = pair['leg1'], pair['leg2']
        
        # Load Data
        df1 = DataManager.load_data(s1, "5m")
        df2 = DataManager.load_data(s2, "5m")
        
        if df1 is None or df2 is None: continue
        
        # Align Data
        df = pd.concat([df1['close'], df2['close']], axis=1).dropna()
        df.columns = ['l1', 'l2']
        
        if len(df) < 200: continue

        # STRATEGY LOGIC: Z-Score Mean Reversion
        # Parameters (Fixed as per instructions: "No Optimization")
        LOOKBACK = 20
        ENTRY_Z = 2.0
        EXIT_Z = 0.5
        
        df['zscore'] = calculate_zscore(df['l1'], df['l2'], window=LOOKBACK)
        
        position = 0 # 0=Flat, 1=Long Spread (Buy L1, Sell L2), -1=Short Spread
        entry_ratio = 0.0
        pnl_points = 0.0
        trades = 0
        
        # Skip warmup
        z_vals = df['zscore'].values
        ratios = (df['l1'] / df['l2']).values
        
        for i in range(LOOKBACK, len(df)):
            z = z_vals[i]
            r = ratios[i]
            
            if np.isnan(z): continue

            # ENTRY LOGIC
            if position == 0:
                if z > ENTRY_Z: 
                    # Spread is High -> SELL Spread (Sell L1, Buy L2)
                    position = -1
                    entry_ratio = r
                    trades += 1
                elif z < -ENTRY_Z:
                    # Spread is Low -> BUY Spread (Buy L1, Sell L2)
                    position = 1
                    entry_ratio = r
                    trades += 1
            
            # EXIT LOGIC
            elif position == -1: # Short
                if z < EXIT_Z: # Reverted to mean
                    pnl_points += (entry_ratio - r) # Short profit if Ratio goes down
                    position = 0
            
            elif position == 1: # Long
                if z > -EXIT_Z: # Reverted to mean
                    pnl_points += (r - entry_ratio) # Long profit if Ratio goes up
                    position = 0

        # Calculate ROI (Approximate based on ratio points)
        # To make it comparable, we assume 1 unit of Ratio traded
        
        if trades > 0:
            results.append({
                "Pair": f"{s1}-{s2}",
                "Leg1": s1,
                "Leg2": s2,
                "Trades": trades,
                "Net PnL": round(pnl_points, 4),
                "P-Value": pair['pvalue']
            })

    # 4. Select Top 3
    if results:
        df_res = pd.DataFrame(results).sort_values(by="Net PnL", ascending=False)
        top_3 = df_res.head(3)
        
        print("\n" + tabulate(top_3, headers="keys", tablefmt="grid"))
        
        # Save Final Config
        final_config = []
        for _, row in top_3.iterrows():
            final_config.append({
                "leg1": row['Leg1'],
                "leg2": row['Leg2'],
                "strategy": "pairs_zscore",
                "lookback": 20,
                "entry_z": 2.0,
                "exit_z": 0.5
            })
            
        with open(config.PAIRS_CONFIG, "w") as f:
            json.dump(final_config, f, indent=4)
            
        print(f"\n✅ Top 3 Pairs saved to {config.PAIRS_CONFIG}")
    else:
        print("\n❌ No profitable pairs found.")

if __name__ == "__main__":
    run_backtest()
