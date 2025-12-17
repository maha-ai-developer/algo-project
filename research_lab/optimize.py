import sys
import os
import itertools
import json
import pandas as pd
import pandas_ta_classic as ta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DATA_DIR = config.DATA_DIR
SYMBOLS_FILE = config.SYMBOLS_FILE
OUTPUT_FILE = config.MOMENTUM_CONFIG

ema_range = [10, 20, 50, 100]
rsi_range = [50, 55, 60, 65]

def optimize_symbol(symbol):
    csv_path = os.path.join(DATA_DIR, f"{symbol}_5m.csv")
    if not os.path.exists(csv_path): return None

    try:
        df = pd.read_csv(csv_path)
    except Exception: return None

    best_pnl = -float('inf')
    best_params = {}

    combinations = list(itertools.product(ema_range, rsi_range))
    total_combos = len(combinations)

    print(f"\n🔍 Optimizing {symbol:<10} ({len(df)} candles)...")

    for idx, (ema_len, rsi_buy) in enumerate(combinations):
        sys.stdout.write(f"\r    Testing Combo {idx+1}/{total_combos}   ")
        sys.stdout.flush()

        sim_df = df.copy()
        sim_df['ema'] = ta.ema(sim_df['close'], length=ema_len)
        sim_df['rsi'] = ta.rsi(sim_df['close'], length=14)
        
        capital = 100000
        position = 0
        closes = sim_df['close'].values
        emas = sim_df['ema'].values
        rsis = sim_df['rsi'].values
        
        if len(closes) < 100: continue

        for i in range(50, len(closes)):
            # BUY
            if position == 0 and closes[i] > emas[i] and rsis[i] > rsi_buy:
                qty = int(capital / closes[i])
                if qty > 0:
                    position = qty
                    capital -= position * closes[i]
            
            # SELL
            elif position > 0 and closes[i] < emas[i]:
                capital += position * closes[i]
                position = 0
        
        final_value = capital
        if position > 0: final_value += position * closes[-1]
        pnl = final_value - 100000
        
        if pnl > best_pnl:
            best_pnl = pnl
            best_params = {"ema": ema_len, "rsi": rsi_buy, "pnl": round(pnl, 2)} # <--- SAVING PNL NOW

    # Filter Losers
    if best_pnl > 0:
        sys.stdout.write(f"\r    ✅ PASS: PnL ₹{best_pnl:,.2f} | EMA {best_params['ema']} | RSI {best_params['rsi']}    \n")
        return best_params
    else:
        sys.stdout.write(f"\r    ❌ FAIL: PnL ₹{best_pnl:,.2f} (Discarding)                    \n")
        return None

if __name__ == "__main__":
    if not os.path.exists(SYMBOLS_FILE):
        print(f"❌ Symbols file not found at: {SYMBOLS_FILE}")
        exit()

    with open(SYMBOLS_FILE, "r") as f:
        symbols = [s.strip().upper() for s in f.readlines() if s.strip()]

    print(f"--- 🚀 OPTIMIZING FOR PROFITABILITY ---")
    
    final_config = {}
    for sym in symbols:
        result = optimize_symbol(sym)
        if result:
            final_config[sym] = result

    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_config, f, indent=4)
        
    print(f"\n✨ DONE! Profitable configurations saved to: {OUTPUT_FILE}")
