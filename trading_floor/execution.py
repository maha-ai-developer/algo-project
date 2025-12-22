import sqlite3
import datetime
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config

# ✅ Import the broker function
from infrastructure.broker.kite_orders import place_order 

class ExecutionHandler:
    def __init__(self, mode="PAPER"):
        self.mode = mode.upper()
        self.db_path = os.path.join(config.DATA_DIR, "trades.db")
        self.init_db()
        print(f"   👮 Execution Handler: {self.mode}")

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                side TEXT,
                quantity INTEGER,
                price REAL,
                strategy TEXT,
                mode TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_trade(self, symbol, side, qty, price, strategy):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, side, quantity, price, strategy, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.datetime.now(), symbol, side, qty, price, strategy, self.mode))
        conn.commit()
        conn.close()

    def get_marketable_limit_price(self, side, current_price, buffer_pct=0.01):
        """
        Calculates a 'Safe' Limit Price.
        BUY:  Current Price + 1% (Willing to pay a bit more to ensure fill)
        SELL: Current Price - 1% (Willing to sell a bit lower to ensure fill)
        """
        if side == "BUY":
            return round(current_price * (1 + buffer_pct), 2)
        else:
            return round(current_price * (1 - buffer_pct), 2)

    def place_pair_order(self, sym1, side1, qty1, px1, sym2, side2, qty2, px2, product="MIS"):
        """
        Executes two legs using MARKETABLE LIMIT ORDERS to protect against slippage.
        """
        # Calculate Safe Limit Prices (1% Buffer)
        limit_px1 = self.get_marketable_limit_price(side1, px1)
        limit_px2 = self.get_marketable_limit_price(side2, px2)

        print(f"      🚀 EXECUTING PAIR (Slippage Prot): {sym1} ({side1} {qty1} @ {limit_px1}) & {sym2} ({side2} {qty2} @ {limit_px2})")
        
        # --- LIVE MODE SWITCH ---
        if self.mode == "LIVE":
            print(f"      📡 SENDING PROTECTED ORDERS TO KITE...")
            
            # 1. Place Leg 1 (Limit Order with Buffer)
            # We use 'LIMIT' type, but the price is aggressive to ensure immediate fill.
            id1 = place_order(sym1, side1, qty1, price=limit_px1, order_type="LIMIT", product=product)
            
            if not id1: 
                print(f"      ❌ Leg 1 ({sym1}) Failed. Aborting Leg 2.")
                return False
            
            # 2. Place Leg 2 (Limit Order with Buffer)
            id2 = place_order(sym2, side2, qty2, price=limit_px2, order_type="LIMIT", product=product)
            
            if not id2:
                print(f"      ❌ Leg 2 ({sym2}) Failed. ⚠️ URGENT: You have an open leg on {sym1}!")
                # Note: In a real HFT system, you would execute a 'Cleanup' trade here to close Leg 1.
                return False
            
            print(f"      ✅ Pair Executed Successfully. IDs: {id1}, {id2}")
            
        else:
            print("      📝 PAPER TRADE LOGGED (No API Call)")

        # Log trades with the INTENDED execution price (px1, px2), not the limit cap
        self.log_trade(sym1, side1, qty1, px1, "StatArb")
        self.log_trade(sym2, side2, qty2, px2, "StatArb")
        return True
