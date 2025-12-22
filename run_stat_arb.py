import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Import your custom modules
from strategies.stat_arb_bot import StatArbBot
from infrastructure.data.data_manager import download_historical_data
import infrastructure.config as config

def run_simulation():
    print("--- 📉 STATISTICAL ARBITRAGE (METHOD 2) SIMULATION ---")

    # 1. SETUP: Define parameters based on the PDF
    # Entry at 2.5 SD, Exit at Mean (0.0), Stop Loss at 3.5 SD
    bot = StatArbBot(entry_z=2.5, exit_z=0.0, stop_z=3.5)

    # 2. SELECT PAIR: Let's test a Banking Sector Pair
    stock_a = "ICICIBANK"
    stock_b = "AXISBANK"
    
    print(f"🔍 Testing Pair: {stock_a} vs {stock_b}")

    # 3. GET DATA: Fetch 1 Year of Daily Data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"⬇️ Downloading Data ({start_date} to {end_date})...")
    download_historical_data([stock_a, stock_b], start_date, end_date, interval="day")

    # Load into DataFrames
    try:
        # CORRECTION: Use config.DATA_DIR instead of HISTORICAL_DIR
        path_a = os.path.join(config.DATA_DIR, f"{stock_a}_day.csv")
        path_b = os.path.join(config.DATA_DIR, f"{stock_b}_day.csv")
        
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print(f"❌ Error: Files not found at {config.DATA_DIR}")
            return

        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
        
        # Set Dates as Index for alignment
        df_a['date'] = pd.to_datetime(df_a['date'])
        df_b['date'] = pd.to_datetime(df_b['date'])
        df_a.set_index('date', inplace=True)
        df_b.set_index('date', inplace=True)
        
        series_a = df_a['close']
        series_b = df_b['close']
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 4. CALIBRATE: Run OLS + Error Ratio Test + ADF
    print("\n⚙️ Running Math (OLS Regression & ADF Test)...")
    is_tradable = bot.calibrate(series_a, series_b, stock_a, stock_b)

    if not is_tradable:
        print("\n❌ PAIR REJECTED.")
        print("   Reason: Residuals are not stationary (P-Value > 0.05).")
        print("   Action: Try a different pair (e.g., HDFCBANK vs KOTAKBANK).")
        return

    # 5. EXECUTION LOGIC
    print("\n✅ PAIR ACCEPTED! Generating Trading Signals...")
    
    # Identify which is Y and X based on the Bot's selection
    if bot.y_symbol == stock_a:
        signals = bot.generate_signals(series_a, series_b)
    else:
        signals = bot.generate_signals(series_b, series_a)

    # 6. REPORTING
    print("-" * 50)
    print(f"🏆 WINNER STRATEGY:")
    print(f"   LONG:  1 Unit of {bot.y_symbol}")
    print(f"   SHORT: {bot.beta:.4f} Units of {bot.x_symbol}")
    print(f"   (Hedge Ratio: {bot.beta:.4f})")
    print("-" * 50)

    # Show only the trades (where position changed)
    signals['trade_action'] = signals['position'].diff()
    trades = signals[signals['trade_action'] != 0].dropna()
    
    if not trades.empty:
        print("\n📜 Recent Trade Signals:")
        print(trades[['z', 'position']].tail(10))
        
        # Simple PnL Calc (Approximate)
        signals['pnl_points'] = signals['position'].shift(1) * (signals['z'] - signals['z'].shift(1))
        total_pnl = signals['pnl_points'].sum()
        
        print(f"\n💰 EST. PERFORMANCE (Z-Score Points Captured): {total_pnl:.4f}")
        if total_pnl > 0:
            print("   ✅ Strategy is PROFITABLE on this data.")
        else:
            print("   ⚠️ Strategy had losses.")
    else:
        print("\n⚠️ No trades triggered (Z-Score never hit 2.5).")

if __name__ == "__main__":
    run_simulation()
