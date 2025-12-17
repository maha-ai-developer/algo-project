import sqlite3
import datetime
import os
import sys

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.broker.kite_orders import place_order

class ExecutionHandler:
    def __init__(self, mode="paper"):
        self.mode = mode
        self.db_path = os.path.join(config.DATA_DIR, "trades.db")
        self.init_db()
        print(f"   👮 Execution Handler Initialized (Mode: {self.mode.upper()})")

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                signal TEXT,
                quantity INTEGER,
                price REAL,
                strategy TEXT,
                mode TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_trade(self, symbol, signal, qty, price, strategy):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, signal, quantity, price, strategy, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.datetime.now(), symbol, signal, qty, price, strategy, self.mode))
        conn.commit()
        conn.close()
        print(f"   📝 Trade Logged: {signal} {qty} {symbol} @ {price}")

    def execute(self, order_dict):
        """
        Executes orders as LIMIT with Buffer to avoid slippage.
        """
        symbol = order_dict['symbol']
        signal = order_dict['signal'] # "BUY" or "SELL"
        qty = order_dict['quantity']
        ltp = order_dict.get('price', 0)
        strategy = order_dict.get('strategy', 'manual')
        tag = order_dict.get('tag', 'algo')

        print(f"\n   🚀 EXECUTION ALERT: {signal} {symbol} (Qty: {qty})")

        if self.mode == "live":
            try:
                # 1. Determine Order Type
                # Panic Exits (Time/SL) -> MARKET (Get out at any cost)
                if strategy in ["TIME_EXIT", "SL_HIT"]:
                    order_type = "MARKET"
                    final_price = 0
                    print(f"   ⚠️ Panic Exit ({strategy}): Using MARKET Order")
                
                # Normal Entries/Exits -> LIMIT (Avoid Slippage)
                else:
                    order_type = "LIMIT"
                    # Add 0.3% Buffer to ensure fill (Market Protection)
                    # Buy Limit = LTP + 0.3% | Sell Limit = LTP - 0.3%
                    if signal == "BUY":
                        final_price = ltp * 1.003
                    else:
                        final_price = ltp * 0.997
                    
                    print(f"   🛡️ Using LIMIT Order: {final_price:.2f} (LTP: {ltp})")

                # 2. Place Order
                # Note: Updated 'transaction_type' -> 'side' to match your uploaded kite_orders.py
                order_id = place_order(
                    symbol=symbol,
                    side=signal, 
                    quantity=qty,
                    price=final_price,
                    order_type=order_type,
                    tag=tag
                )

                if order_id:
                    print(f"   ✅ LIVE Order Placed! ID: {order_id}")
                    self.log_trade(symbol, signal, qty, ltp, strategy)
            
            except Exception as e:
                print(f"   ❌ LIVE Execution Failed: {e}")

        else:
            # PAPER TRADE
            print(f"   📄 [PAPER] Simulated {signal} {symbol} @ {ltp:.2f}")
            self.log_trade(symbol, signal, qty, ltp, strategy)
