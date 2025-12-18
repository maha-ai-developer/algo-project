import pandas as pd
import os
import sys
import json
import itertools
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.data_manager import download_historical_data

def scan_pairs():
    print("--- ⚖️ PHASE 3: SECTOR-AWARE PAIR SCANNER ---")

    # 1. Load Fundamental Winners
    fundamental_path = os.path.join(config.ARTIFACTS_DIR, "fundamental_analysis.csv")
    if not os.path.exists(fundamental_path):
        print(f"❌ Error: {fundamental_path} not found. Run scan_fundamental first.")
        return

    df_fund = pd.read_csv(fundamental_path)
    
    # Check if 'Sector' column exists (migration check)
    if 'Sector' not in df_fund.columns:
        print("❌ 'Sector' column missing in CSV. Please re-run scan_fundamental.")
        return

    # FILTER: Only Investible Stocks
    df_investible = df_fund[df_fund['Quality'] == 'INVESTIBLE'].copy()
    
    if len(df_investible) < 2:
        print("❌ Not enough 'INVESTIBLE' stocks to form pairs.")
        return

    # 2. Group Stocks by Sector
    # We only want to test pairs WITHIN the same sector.
    sector_groups = df_investible.groupby('Sector')['Symbol'].apply(list).to_dict()
    
    all_symbols = df_investible['Symbol'].unique().tolist()
    print(f"💎 Universe: {len(all_symbols)} Investible Stocks.")
    print(f"📂 Sectors Found: {list(sector_groups.keys())}")

    # 3. Download DAILY Data (1 Year)
    print("\n⬇️ Fetching Training Data (365 Days Daily)...")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    download_historical_data(all_symbols, from_date, to_date, interval="day")

    # 4. Load Data into Memory
    price_data = {}
    for symbol in all_symbols:
        try:
            path = os.path.join(config.HISTORICAL_DIR, f"{symbol}_day.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                price_data[symbol] = df['close']
        except: continue

    # 5. Run Cointegration Tests (Grouped by Sector)
    print("\n🔬 Running Tests (Sector-Matched Only)...")
    pairs_candidates = []
    
    for sector, symbols in sector_groups.items():
        if len(symbols) < 2: continue # Need at least 2 stocks in a sector to pair
        
        # Generate combinations ONLY within this sector
        sector_pairs = list(itertools.combinations(symbols, 2))
        
        for s1, s2 in sector_pairs:
            # Align Data
            if s1 not in price_data or s2 not in price_data: continue
            
            s1_prices = price_data[s1]
            s2_prices = price_data[s2]
            df = pd.concat([s1_prices, s2_prices], axis=1).dropna()
            
            if len(df) < 100: continue 

            # Statistical Test
            score, pvalue, _ = coint(df.iloc[:,0], df.iloc[:,1])
            corr = df.iloc[:,0].corr(df.iloc[:,1])
            
            # Criteria: High Correlation + Cointegration
            if pvalue < 0.05 and corr > 0.80:
                print(f"   🎯 FOUND [{sector}]: {s1}-{s2} | P-Val: {pvalue:.4f} | Corr: {corr:.2f}")
                
                pairs_candidates.append({
                    "leg1": s1, "leg2": s2,
                    "sector": sector,
                    "correlation": round(corr, 2),
                    "pvalue": round(pvalue, 6)
                })

    # 6. Save Results
    if pairs_candidates:
        with open(config.PAIRS_CANDIDATES_FILE, "w") as f:
            json.dump(pairs_candidates, f, indent=4)
        print(f"\n✅ Found {len(pairs_candidates)} Valid Pairs.")
    else:
        print("\n❌ No valid pairs found.")

if __name__ == "__main__":
    scan_pairs()
