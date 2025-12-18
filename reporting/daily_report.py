import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime
from tabulate import tabulate

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config

def generate_daily_report():
    print(f"\n📊 --- DAILY TRADING REPORT: {datetime.now().strftime('%Y-%m-%d')} ---")
    
    db_path = os.path.join(config.DATA_DIR, "trades.db")
    if not os.path.exists(db_path):
        print("❌ No trade database found.")
        return

    conn = sqlite3.connect(db_path)
    
    # Fetch trades for TODAY only
    query = """
        SELECT timestamp, symbol, signal, quantity, price, strategy, mode 
        FROM trades 
        WHERE date(timestamp) = date('now', 'localtime')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("💤 No trades executed today.")
        return

    # --- 1. PERFORMANCE METRICS ---
    summary = []
    total_pnl = 0.0
    total_brokerage = 0.0
    
    # Group by Symbol to calculate PnL per stock
    symbols = df['symbol'].unique()
    
    for sym in symbols:
        sym_trades = df[df['symbol'] == sym]
        
        buys = sym_trades[sym_trades['signal'] == 'BUY']
        sells = sym_trades[sym_trades['signal'] == 'SELL']
        
        buy_qty = buys['quantity'].sum()
        sell_qty = sells['quantity'].sum()
        
        avg_buy = (buys['price'] * buys['quantity']).sum() / buy_qty if buy_qty > 0 else 0
        avg_sell = (sells['price'] * sells['quantity']).sum() / sell_qty if sell_qty > 0 else 0
        
        # Calculate Realized PnL (Matched Qty only)
        matched_qty = min(buy_qty, sell_qty)
        gross_pnl = (avg_sell - avg_buy) * matched_qty
        
        # Estimated Brokerage (Zerodha approx: 0.03% or Rs 20, whichever is lower per order)
        # Simplified: 0.05% of turnover for conservative est
        turnover = (avg_buy * matched_qty) + (avg_sell * matched_qty)
        est_charges = turnover * 0.0005 
        
        net_pnl = gross_pnl - est_charges
        
        status = "OPEN" if buy_qty != sell_qty else "CLOSED"
        
        summary.append({
            "Symbol": sym,
            "Trades": len(sym_trades),
            "Status": status,
            "Realized PnL": round(gross_pnl, 2),
            "Charges (Est)": round(est_charges, 2),
            "Net PnL": round(net_pnl, 2)
        })
        
        total_pnl += gross_pnl
        total_brokerage += est_charges

    # --- 2. DISPLAY TABLE ---
    df_summary = pd.DataFrame(summary)
    print("\n" + tabulate(df_summary, headers="keys", tablefmt="simple_grid"))

    # --- 3. FINAL SCORECARD ---
    net_total = total_pnl - total_brokerage
    win_count = len(df_summary[df_summary['Net PnL'] > 0])
    loss_count = len(df_summary[df_summary['Net PnL'] <= 0])
    win_rate = (win_count / len(df_summary)) * 100 if len(df_summary) > 0 else 0

    print(f"\n💰 GROSS PnL:   ₹ {total_pnl:,.2f}")
    print(f"📉 CHARGES:     ₹ {total_brokerage:,.2f}")
    print(f"💵 NET PnL:     ₹ {net_total:,.2f}")
    print(f"---------------------------------")
    print(f"🎯 Win Rate:    {win_rate:.1f}% ({win_count} Wins / {loss_count} Losses)")
    
    if net_total > 0:
        print("🚀 RESULT:      PROFITABLE DAY")
    else:
        print("🔻 RESULT:      LOSS DAY")

if __name__ == "__main__":
    generate_daily_report()
