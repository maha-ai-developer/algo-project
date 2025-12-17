import pandas as pd
import os
import sys
import json
import itertools
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.data_manager import download_historical_data, DataManager

def scan_pairs():
    print("--- ⚖️ PHASE 3: QUALITY PAIR SCANNER ---")

    # 1. Load Fundamental Winners
    if not os.path.exists(config.FUNDAMENTAL_FILE):
        print(f"❌ Error: {config.FUNDAMENTAL_FILE} not found. Run Phase 1 first.")
        return

    df_fund = pd.read_csv(config.FUNDAMENTAL_FILE)
    
    # FILTER: Only Fundamental Champions
    investible_stocks = df_fund[df_fund['Quality'] == 'INVESTIBLE']['Symbol'].tolist()
    
    if len(investible_stocks) < 2:
        print("❌ Not enough 'INVESTIBLE' stocks to form pairs.")
        return

    print(f"💎 Fundamental Universe: {len(investible_stocks)} stocks")

    # 2. Download DAILY Data (365 Days)
    print("\n⬇️ Fetching 1 Year Daily Data for Cointegration Test...")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    download_historical_data(investible_stocks, from_date, to_date, interval="day")

    # 3. Load Data into Memory
    price_data = pd.DataFrame()
    for sym in investible_stocks:
        df = DataManager.load_data(sym, "day")
        if df is not None and not df.empty:
            # Reindex to ensure matching dates
            price_data[sym] = df['close']
    
    # Drop rows with NaN (alignment)
    price_data.dropna(inplace=True)
    
    if price_data.shape[1] < 2:
        print("❌ Not enough aligned data to scan pairs.")
        return

    print(f"🧮 Scanning {len(price_data.columns)} stocks over {len(price_data)} trading days...")

    # 4. Statistical Tests (Correlation + Cointegration)
    pairs_candidates = []
    unique_symbols = set()
    
    # Generate all unique pairs
    keys = price_data.columns
    for s1, s2 in itertools.combinations(keys, 2):
        
        # A. Correlation Check (Fast Filter)
        corr = price_data[s1].corr(price_data[s2])
        if corr < config.PAIR_CORRELATION_MIN: 
            continue

        # B. Cointegration Test (The Real Deal)
        # Null Hypothesis: Spread is Non-Stationary (Random Walk)
        # We want to REJECT Null (P-Value < 0.05)
        score, pvalue, _ = coint(price_data[s1], price_data[s2])
        
        if pvalue < config.PAIR_PVALUE_MAX:
            print(f"   🎯 FOUND: {s1}-{s2} | Corr: {corr:.2f} | P-Val: {pvalue:.5f}")
            
            # Calculate Static Spread Params for Reference
            ratio = price_data[s1] / price_data[s2]
            
            pairs_candidates.append({
                "leg1": s1,
                "leg2": s2,
                "correlation": round(corr, 2),
                "pvalue": round(pvalue, 6),
                "z_mean": round(ratio.mean(), 4),
                "z_std": round(ratio.std(), 4)
            })
            unique_symbols.add(s1)
            unique_symbols.add(s2)

    # 5. Save Results
    if pairs_candidates:
        # Save Logic JSON
        with open(config.PAIRS_CANDIDATES_FILE, "w") as f:
            json.dump(pairs_candidates, f, indent=4)
            
        # Save Simple List for Phase 3b (Download)
        pairs_txt_path = os.path.join(config.ARTIFACTS_DIR, "pairs_dataset.txt")
        with open(pairs_txt_path, "w") as f:
            for s in unique_symbols:
                f.write(f"{s}\n")
                
        print(f"\n✅ Found {len(pairs_candidates)} Cointegrated Pairs.")
        print(f"📄 Candidates saved to: {config.PAIRS_CANDIDATES_FILE}")
        print(f"📄 Symbol list saved to: {pairs_txt_path}")
        print("🚀 Ready for Backtesting (5-min data).")
    else:
        print("❌ No pairs passed the statistical tests.")

if __name__ == "__main__":
    scan_pairs()
