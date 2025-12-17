import pandas as pd
import json
import os
from datetime import datetime, date
from tabulate import tabulate
from sqlalchemy import cast, Date

# 1. IMPORTS FOR DB CONNECTION
from db.pg import get_session, Trade, init_db

def load_db_url():
    """
    Reads the DB URL directly from config/config.json
    """
    try:
        # Path to your config.json
        config_path = os.path.join("config", "config.json")
        
        if not os.path.exists(config_path):
            print(f"[Warn] {config_path} not found. Trying default SQLite.")
            return "sqlite:///algo_trading.db"
            
        with open(config_path, "r") as f:
            data = json.load(f)
            # navigate to db -> url
            return data.get("db", {}).get("url", "sqlite:///algo_trading.db")
            
    except Exception as e:
        print(f"[Error] Reading config.json failed: {e}")
        return "sqlite:///algo_trading.db"

def generate_daily_report():
    print("\n--- 📊 GENERATING DAILY TRADING REPORT ---")
    
    # 2. GET URL & CONNECT
    db_url = load_db_url()
    print(f"Connecting to Database...") # Hiding credentials for security
    
    try:
        init_db(db_url)
    except Exception as e:
        print(f"[Critical Error] Database connection failed.")
        print(f"Check your config/config.json. Current URL: {db_url}")
        print(f"Details: {e}")
        return

    session = get_session()
    today = date.today()
    
    print(f"Fetching trades for Date: {today}")

    # 3. Query Database
    try:
        trades = session.query(Trade).filter(
            cast(Trade.timestamp, Date) == today
        ).all()
    except Exception as e:
        print(f"[Error] SQL Query failed: {e}")
        return

    if not trades:
        print("\n[Result] No trades were executed today.")
        print("This is not bad! It means the market did not give a clear 100% signal.")
        print("Capital Preserved: ✅")
        return

    # 4. Convert to DataFrame
    data = []
    for t in trades:
        data.append({
            "Time": t.timestamp.strftime("%H:%M:%S"),
            "Symbol": t.symbol,
            "Side": t.side,
            "Qty": t.qty,
            "Price": t.price,
            "PnL": t.pnl if t.pnl else 0.0,
            "Status": "REAL" if not t.sim else "SIM"
        })

    df = pd.DataFrame(data)

    # 5. Summary Metrics
    total_trades = len(df)
    total_pnl = df['PnL'].sum()
    winners = len(df[df['PnL'] > 0])
    losers = len(df[df['PnL'] < 0])
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

    # 6. Display Report
    print(f"\nSummary for {today}:")
    print(f"---------------------------")
    print(f"Total P&L    : ₹ {total_pnl:.2f}")
    print(f"Total Trades : {total_trades}")
    print(f"Win Rate     : {win_rate:.1f}%")
    print(f"---------------------------")

    print("\nTrade Log:")
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    # 7. Save to CSV
    filename = f"report_{today}.csv"
    df.to_csv(filename, index=False)
    print(f"\n[Saved] Detailed report saved to: {filename}")

if __name__ == "__main__":
    generate_daily_report()
