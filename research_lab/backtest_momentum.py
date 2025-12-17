import pandas as pd
import pandas_ta_classic as ta
import os
import sys
import json
import itertools
from datetime import datetime, timedelta
from tabulate import tabulate

# Add root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.data_manager import download_historical_data, DataManager

# --- OPTIMIZATION SETTINGS ---
PARAM_GRID = {
    "ema_period": [20, 50, 100, 200],      # Fast vs Slow Trends
    "rsi_entry": [55, 60, 65, 70],         # Aggressive vs Conservative
    "rsi_exit": [40]                       # Standard exit (can be optimized too)
}

def run_backtest():
    print("--- 🏎️ PHASE 2b: MOMENTUM OPTIMIZATION (GRID SEARCH) ---")
    
    # 1. Load Candidates from Phase 2a
    candidate_file = os.path.join(config.ARTIFACTS_DIR, "momentum_candidates.txt")
    if not os.path.exists(candidate_file):
        print("❌ Candidate file missing. Run scan_momentum.py first.")
        return
        
    with open(candidate_file, "r") as f:
        symbols = [s.strip() for s in f.readlines() if s.strip()]

    # 2. Ensure Data Exists
    print(f"🔄 Checking 5-min data for {len(symbols)} candidates...")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    download_historical_data(symbols, from_date, to_date, interval="5m")

    final_config = {}
    report_data = []

    # 3. Optimization Loop
    for sym in symbols:
        print(f"\n🔬 Optimizing {sym}...", end="")
        df_raw = DataManager.load_data(sym, "5m")
        if df_raw is None: 
            print(" (No Data)")
            continue

        best_pnl = -999.0
        best_params = {}
        best_trades = 0

        # Generate all combinations of parameters
        keys, values = zip(*PARAM_GRID.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # --- GRID SEARCH ---
        for params in combinations:
            # Create a copy to avoid overwriting logic
            df = df_raw.copy()
            
            # Calculate Indicators
            ema_len = params['ema_period']
            df[f'ema'] = ta.ema(df['close'], length=ema_len)
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # Vectorized Backtest (Simplified for Speed)
            # Signal: 1 = BUY, 0 = FLAT
            df['signal'] = 0
            
            # Entry Condition
            buy_cond = (df['rsi'] > params['rsi_entry']) & (df['close'] > df['ema'])
            
            # Exit Condition
            sell_cond = (df['rsi'] < params['rsi_exit']) | (df['close'] < df['ema'])
            
            # Simulation Logic
            position = 0
            entry_price = 0.0
            total_pnl = 0.0
            trade_count = 0
            
            # Iterate (Vectorized is harder with stateful exit, using fast loop)
            # Skip warmup
            prices = df['close'].values
            emas = df['ema'].values
            rsis = df['rsi'].values
            
            for i in range(200, len(prices)):
                price = prices[i]
                
                if position == 0:
                    # ENTRY
                    if rsis[i] > params['rsi_entry'] and price > emas[i]:
                        position = 1
                        entry_price = price
                        trade_count += 1
                
                elif position == 1:
                    # EXIT
                    if rsis[i] < params['rsi_exit'] or price < emas[i]:
                        position = 0
                        pct_change = (price - entry_price) / entry_price
                        total_pnl += pct_change

            # Check if this is the best combo
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_params = params
                best_trades = trade_count
                print(".", end="", flush=True) # Progress dot

        # Store Winner
        if best_pnl > 0: # Only accept profitable strategies
            print(f" ✅ Best: EMA{best_params['ema_period']} / RSI{best_params['rsi_entry']} (PnL: {best_pnl*100:.1f}%)")
            
            final_config[sym] = {
                "strategy": "momentum_optimized",
                "ema_period": best_params['ema_period'],
                "rsi_entry": best_params['rsi_entry'],
                "rsi_exit": best_params['rsi_exit'],
                "expected_pnl": round(best_pnl * 100, 2)
            }
            
            report_data.append({
                "Symbol": sym,
                "Best EMA": best_params['ema_period'],
                "Best RSI": best_params['rsi_entry'],
                "Trades": best_trades,
                "PnL %": round(best_pnl * 100, 2)
            })
        else:
            print(" ❌ No profitable params found.")

    # 4. Final Report & Save
    if report_data:
        df_res = pd.DataFrame(report_data).sort_values(by="PnL %", ascending=False)
        print("\n" + tabulate(df_res, headers="keys", tablefmt="grid"))
        
        with open(config.MOMENTUM_CONFIG, "w") as f:
            json.dump(final_config, f, indent=4)
        print(f"\n✅ Optimization saved to {config.MOMENTUM_CONFIG}")
    else:
        print("\n❌ No candidates passed optimization.")

if __name__ == "__main__":
    run_backtest()
