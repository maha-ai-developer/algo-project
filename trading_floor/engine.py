import time
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infrastructure.config as config
from infrastructure.broker.kite_auth import get_kite
from trading_floor.execution import ExecutionHandler
from trading_floor.risk_manager import RiskManager
from strategies.pairs import PairStrategy

# --- ⚙️ USER SETTINGS ---
# "MIS"  = Intraday (5x Leverage). Allows Shorting.
# "CNC"  = Delivery (1x Leverage). No Shorting.
PRODUCT_TYPE = "MIS" 

# --- 💰 CAPITAL SETTING ---
# Set to ₹50,000 as requested
TOTAL_CAPITAL = 50000 

class TradingEngine:
    def __init__(self, mode="paper"):
        self.mode = mode.upper()
        print(f"\n--- 🚀 STAT ARB ENGINE ({self.mode}) ---")
        print(f"   ⚙️ Product Type: {PRODUCT_TYPE}")
        print(f"   💰 Capital Base: ₹{TOTAL_CAPITAL:,}")
        
        # 1. Initialize Components
        self.kite = get_kite()
        self.executor = ExecutionHandler(mode=self.mode)
        
        # Risk Manager configured with your capital
        self.risk_manager = RiskManager(capital_per_pair=TOTAL_CAPITAL)
        
        # 2. Load Pair Configuration
        if not os.path.exists(config.PAIRS_CONFIG):
            print(f"❌ Config not found: {config.PAIRS_CONFIG}")
            sys.exit(1)
            
        with open(config.PAIRS_CONFIG, "r") as f:
            self.pairs_config = json.load(f)
            
        # 3. Cache Instrument Tokens (Essential for Live Data)
        print("📊 Fetching Instrument Tokens from Zerodha...")
        self.tokens = {}
        try:
            self.instruments = self.kite.instruments("NSE")
            self.inst_map = {i['tradingsymbol']: i['instrument_token'] for i in self.instruments}
        except Exception as e:
            print(f"   ⚠️ Failed to fetch instruments: {e}")
            self.inst_map = {}
        
        # 4. Initialize Strategy Brains
        self.strategies = {}
        self.active_trades = {} 
        
        for p in self.pairs_config:
            pair_key = f"{p['leg1']}-{p['leg2']}"
            
            # Cache Tokens
            if p['leg1'] in self.inst_map: self.tokens[p['leg1']] = self.inst_map[p['leg1']]
            if p['leg2'] in self.inst_map: self.tokens[p['leg2']] = self.inst_map[p['leg2']]
            
            print(f"   🧠 Loaded Agent: {pair_key:<20} (Beta: {p['hedge_ratio']:.2f})")
            
            self.strategies[pair_key] = PairStrategy(
                hedge_ratio=p['hedge_ratio'],
                intercept=p['intercept']
            )

    def get_market_data(self, symbol):
        """Fetches LIVE Market Data (Daily Candles)"""
        if symbol not in self.tokens:
            print(f"   ⚠️ Token missing for {symbol}")
            return pd.Series()

        token = self.tokens[symbol]
        to_date = datetime.now()
        from_date = to_date - timedelta(days=120) 

        try:
            data = self.kite.historical_data(token, from_date, to_date, "day")
            if not data: return pd.Series()
            df = pd.DataFrame(data)
            return df['close']
        except Exception as e:
            print(f"   ⚠️ API Error for {symbol}: {e}")
            return pd.Series()

    def run(self):
        print(f"\n✅ Engine Running... Monitoring {len(self.pairs_config)} Pairs.")
        print("   (Press Ctrl+C to Stop)\n")
        try:
            while True:
                print(f"⏰ Heartbeat: {datetime.now().strftime('%H:%M:%S')}")
                
                for p in self.pairs_config:
                    self.process_pair(p)
                    time.sleep(0.5) 
                    
                time.sleep(60) 
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping Engine.")

    def process_pair(self, p):
        s1, s2 = p['leg1'], p['leg2']
        pair_key = f"{s1}-{s2}"
        strategy = self.strategies[pair_key]
        
        # 1. Fetch Live Data
        data_y = self.get_market_data(s1)
        data_x = self.get_market_data(s2)
        
        if len(data_y) < 60 or len(data_x) < 60: 
            return

        # --- ✅ NEW SAFETY FILTER ---
        # Check if the latest price is valid (> 0 and not NaN)
        last_y = data_y.iloc[-1]
        last_x = data_x.iloc[-1]
        
        if last_y <= 0 or last_x <= 0 or pd.isna(last_y) or pd.isna(last_x):
            print(f"   🛡️ BLOCKED BAD DATA for {pair_key}: {s1}={last_y}, {s2}={last_x}")
            return # Skip this cycle, do not poison the Guardian
        # ----------------------------
        
        # Capture Latest Prices for Execution
        current_price_y = last_y
        current_price_x = last_x

        # 2. Ask Strategy
        response = strategy.generate_signal(data_y, data_x)
        
        health = response.get('health', 'UNKNOWN')
        z = response.get('zscore', 0.0)
        signal = response['signal']
        
        print(f"   👉 {pair_key:<20} | Health: {health:<6} | Z: {z:>5.2f} | Sig: {signal}")

        # 3. Risk Checks (Now passing Prices to close_position)
        if health == "RED":
            if pair_key in self.active_trades:
                print(f"   🔴 GUARDIAN KILL: Closing {pair_key} ({response.get('health_reason')})")
                self.close_position(pair_key, s1, s2, current_price_y, current_price_x)
            return

        is_stop, stop_msg = self.risk_manager.check_stop_loss(z, stop_z_threshold=4.0)
        if is_stop and pair_key in self.active_trades:
            print(f"   🟠 STOP LOSS: {stop_msg}")
            self.close_position(pair_key, s1, s2, current_price_y, current_price_x)
            return

        is_tp, tp_msg = self.risk_manager.check_take_profit(z)
        if is_tp and pair_key in self.active_trades:
            print(f"   🟢 TAKE PROFIT: {tp_msg}")
            self.close_position(pair_key, s1, s2, current_price_y, current_price_x)
            return

        # 4. Entry
        if pair_key not in self.active_trades:
            qty_y, qty_x = self.risk_manager.calculate_sizing(current_price_y, current_price_x, p['hedge_ratio'])
            
            if qty_y == 0 or qty_x == 0: return

            if signal == "LONG_SPREAD":
                print(f"   🚀 ENTRY LONG: Buy {qty_y} {s1}, Sell {qty_x} {s2}")
                self.executor.place_pair_order(
                    s1, "BUY", qty_y, current_price_y,
                    s2, "SELL", qty_x, current_price_x, 
                    product=PRODUCT_TYPE
                )
                self.active_trades[pair_key] = {"side": "LONG", "q1": qty_y, "q2": qty_x}
                
            elif signal == "SHORT_SPREAD":
                print(f"   🚀 ENTRY SHORT: Sell {qty_y} {s1}, Buy {qty_x} {s2}")
                self.executor.place_pair_order(
                    s1, "SELL", qty_y, current_price_y,
                    s2, "BUY", qty_x, current_price_x,
                    product=PRODUCT_TYPE
                )
                self.active_trades[pair_key] = {"side": "SHORT", "q1": qty_y, "q2": qty_x}

    def close_position(self, pair_key, s1, s2, px1, px2):
        if pair_key not in self.active_trades: return
        trade = self.active_trades[pair_key]
        print(f"   📉 CLOSING POSITION: {pair_key}")
        
        if trade['side'] == "LONG":
            self.executor.place_pair_order(
                s1, "SELL", trade['q1'], px1,
                s2, "BUY", trade['q2'], px2,
                product=PRODUCT_TYPE
            )
        else:
            self.executor.place_pair_order(
                s1, "BUY", trade['q1'], px1,
                s2, "SELL", trade['q2'], px2,
                product=PRODUCT_TYPE
            )
            
        del self.active_trades[pair_key]

if __name__ == "__main__":
    pass
