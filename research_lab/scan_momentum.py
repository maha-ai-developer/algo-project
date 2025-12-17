import pandas as pd
import pandas_ta_classic as ta
import os
import sys
import json
from datetime import datetime, timedelta

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.data_manager import download_historical_data, DataManager

def scan_momentum():
    print("--- 🚀 PHASE 2: HYBRID MOMENTUM SCAN ---")

    # 1. Load Fundamental Winners
    if not os.path.exists(config.FUNDAMENTAL_FILE):
        print(f"❌ Error: {config.FUNDAMENTAL_FILE} not found. Run Phase 1 first.")
        return

    df_fund = pd.read_csv(config.FUNDAMENTAL_FILE)
    
    # FILTER: strict quality check
    investible_stocks = df_fund[df_fund['Quality'] == 'INVESTIBLE']['Symbol'].tolist()
    
    if not investible_stocks:
        print("❌ No 'INVESTIBLE' stocks found in fundamental analysis.")
        return

    print(f"💎 Fundamental Survivors: {len(investible_stocks)} stocks")
    print(f"   {', '.join(investible_stocks[:5])}...")

    # 2. Download DAILY Data for Calculation (365 Days)
    print("\n⬇️ Fetching 1 Year Daily Data for Momentum Calculation...")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Batch Download
    download_historical_data(investible_stocks, from_date, to_date, interval="day")

    # 3. Calculate Momentum
    momentum_scores = []
    
    print("\n🧮 Calculating Momentum Scores...")
    for sym in investible_stocks:
        df = DataManager.load_data(sym, "day")
        if df is None or len(df) < 200:
            continue

        # --- MOMENTUM FORMULA (Ref: Trading System) ---
        # 1. Trend Filter: Price > 200 EMA
        df['ema200'] = ta.ema(df['close'], length=200)
        
        # 2. ROC (Rate of Change) - 6 Month (125 days)
        df['roc125'] = ta.roc(df['close'], length=125)
        
        # 3. Volatility (ATR)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        current = df.iloc[-1]
        
        # LOGIC: Must be in Uptrend AND have positive momentum
        if current['close'] > current['ema200']:
            score = current['roc125'] # Simple ROC ranking
            
            momentum_scores.append({
                "symbol": sym,
                "score": round(score, 2),
                "close": current['close'],
                "atr": round(current['atr'], 2)
            })

    # 4. Rank & Select Top 10
    if not momentum_scores:
        print("❌ No stocks passed the Trend Filter (Price > 200 EMA). Market might be Bearish.")
        return

    df_mom = pd.DataFrame(momentum_scores).sort_values(by='score', ascending=False)
    
    print("\n🏆 TOP MOMENTUM CANDIDATES:")
    print(df_mom.head(10).to_string(index=False))

    # Save to JSON for Backtester/Engine
    top_picks = df_mom.head(10)['symbol'].tolist()
    
    # Save simple list for Phase 2b (Backtest)
    momentum_list_file = os.path.join(config.ARTIFACTS_DIR, "momentum_candidates.txt")
    with open(momentum_list_file, "w") as f:
        for s in top_picks:
            f.write(f"{s}\n")
            
    print(f"\n✅ Saved Top {len(top_picks)} candidates to {momentum_list_file}")
    print("🚀 Ready for Backtesting (5-min data).")

if __name__ == "__main__":
    scan_momentum()
