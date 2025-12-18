import time
import json
import os
import sys
import pandas as pd
import pandas_ta_classic as ta
import numpy as np
from datetime import datetime
from tabulate import tabulate

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infrastructure.config as config
from infrastructure.broker.kite_auth import get_kite
from infrastructure.data.instrument_cache import get_instrument_token
from trading_floor.execution import ExecutionHandler
from trading_floor.risk_manager import RiskManager

# --- 🎛️ STRATEGY TOGGLES ---
RUN_MOMENTUM = False
RUN_PAIRS = True

# --- 💰 CAPITAL ALLOCATION (UPDATED) ---
CAPITAL_PER_LEG = 5000  # <--- CHANGED TO ₹5,000

class TradingEngine:
    def __init__(self, mode="paper"):
        self.mode = mode
        self.kite = get_kite()
        self.executor = ExecutionHandler(mode=mode)
        
        # Risk System (Capital: 50k for margin safety)
        self.risk_manager = RiskManager(total_capital=50000, max_risk_pct=0.02)

        self.momentum_config = self.load_json(config.MOMENTUM_CONFIG)
        self.pairs_config = self.load_json(config.PAIRS_CONFIG)
        
        self.last_processed_candle = {} 
        self.open_positions = {} 
        self.active_risk_state = {} 

        print(f"\n✨ ENGINE STARTED in {mode.upper()} mode")
        print(f"   💰 Sizing Model: Dollar Neutral (Target ₹{CAPITAL_PER_LEG} per leg)")

    def load_json(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: return json.load(f)
            except: return {}
        return {} if path == config.MOMENTUM_CONFIG else []

    def fetch_latest_candles(self, symbol, interval="5minute", lookback_days=5):
        try:
            token = get_instrument_token(symbol)
            if not token: return None
            to_date = datetime.now()
            from_date = to_date - pd.Timedelta(days=lookback_days)
            records = self.kite.historical_data(token, from_date, to_date, interval)
            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
        except: pass
        return None

    def print_portfolio_status(self):
        """Prints live dashboard."""
        if not self.open_positions: return

        status_data = []
        for key, side in self.open_positions.items():
            if '-' in key: # Pair
                display_side = side['side'] if isinstance(side, dict) else side
                status_data.append({"Asset": key, "Type": "PAIR", "Side": display_side, "Status": "Active"})
                continue
            
            risk = self.active_risk_state.get(key, {})
            entry = risk.get('entry', 0)
            status_data.append({"Asset": key, "Type": "MOMENTUM", "Side": side, "Entry": entry})
            
        print("\n" + tabulate(status_data, headers="keys", tablefmt="simple_grid"))

    def run_momentum_strategy(self):
        pass 

    # =================================================================
    # PAIRS STRATEGY (UPDATED: VALUE MATCHING)
    # =================================================================
    def run_pairs_strategy(self):
        if not RUN_PAIRS: return

        for pair_cfg in self.pairs_config:
            s1 = pair_cfg['leg1']
            s2 = pair_cfg['leg2']
            pair_key = f"{s1}-{s2}"
            
            if s1 in self.open_positions or s2 in self.open_positions: continue
            
            time.sleep(0.2)
            df1 = self.fetch_latest_candles(s1, "5minute")
            df2 = self.fetch_latest_candles(s2, "5minute")
            if df1 is None or df2 is None: continue
            
            df = pd.concat([df1['close'], df2['close']], axis=1).dropna()
            df.columns = ['l1', 'l2']
            if df.empty: continue
            
            last_t = df.index[-1]
            if last_t.tzinfo: last_t = last_t.tz_localize(None)
            if (datetime.now() - last_t).total_seconds() > 900: continue

            # Z-Score Calculation
            ratio = df['l1'] / df['l2']
            zscore = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std()
            curr_z = zscore.iloc[-1]
            
            # --- CONFIGURATION (STRICT) ---
            entry_z = 2.5  
            exit_z = 0.0   

            # --- DYNAMIC SIZING CALCULATOR ---
            price_1 = df['l1'].iloc[-1]
            price_2 = df['l2'].iloc[-1]
            
            # Use the global CAPITAL_PER_LEG
            qty_1 = int(CAPITAL_PER_LEG / price_1)
            qty_2 = int(CAPITAL_PER_LEG / price_2)
            
            # Minimum safety
            qty_1 = max(1, qty_1)
            qty_2 = max(1, qty_2)

            # ENTRY LOGIC
            if pair_key not in self.open_positions:
                
                if curr_z > entry_z: # Short Spread (Sell S1, Buy S2)
                    print(f"   ⚡ PAIR ENTER: {pair_key} (Short) | Val: ~{int(qty_1*price_1)} / ~{int(qty_2*price_2)}")
                    s1_ok = self.executor.execute({"symbol": s1, "signal": "SELL", "quantity": qty_1, "price": price_1, "strategy": "pair"})
                    s2_ok = self.executor.execute({"symbol": s2, "signal": "BUY", "quantity": qty_2, "price": price_2, "strategy": "pair"})
                    
                    if s1_ok and s2_ok:
                        self.open_positions[pair_key] = {'side': 'SHORT', 'q1': qty_1, 'q2': qty_2}
                
                elif curr_z < -entry_z: # Long Spread (Buy S1, Sell S2)
                    print(f"   ⚡ PAIR ENTER: {pair_key} (Long) | Val: ~{int(qty_1*price_1)} / ~{int(qty_2*price_2)}")
                    s1_ok = self.executor.execute({"symbol": s1, "signal": "BUY", "quantity": qty_1, "price": price_1, "strategy": "pair"})
                    s2_ok = self.executor.execute({"symbol": s2, "signal": "SELL", "quantity": qty_2, "price": price_2, "strategy": "pair"})
                    
                    if s1_ok and s2_ok:
                        self.open_positions[pair_key] = {'side': 'LONG', 'q1': qty_1, 'q2': qty_2}

            # EXIT LOGIC
            elif pair_key in self.open_positions:
                pos_data = self.open_positions[pair_key]
                state = pos_data['side']
                q1 = pos_data['q1']
                q2 = pos_data['q2']
                
                if abs(curr_z) < exit_z:
                    print(f"   ⚡ PAIR EXIT: {pair_key} (Mean Reversion)")
                    sig1 = "BUY" if state == 'SHORT' else "SELL"
                    sig2 = "SELL" if state == 'SHORT' else "BUY"
                    
                    s1_ok = self.executor.execute({"symbol": s1, "signal": sig1, "quantity": q1, "price": price_1, "strategy": "pair_exit"})
                    s2_ok = self.executor.execute({"symbol": s2, "signal": sig2, "quantity": q2, "price": price_2, "strategy": "pair_exit"})
                    
                    if s1_ok and s2_ok:
                        del self.open_positions[pair_key]

    def close_position(self, symbol, price, reason):
        if symbol in self.open_positions:
            self.executor.execute({
                "symbol": symbol, "signal": "SELL", "quantity": 1,
                "price": price, "strategy": reason
            })
            del self.open_positions[symbol]

    def check_global_exits(self):
        if self.risk_manager.check_time_exit():
            if self.open_positions:
                print("\n   ⏰ MARKET CLOSING - AUTO SQUARE OFF")
                for key, data in list(self.open_positions.items()):
                    if '-' in key: # PAIR
                        s1, s2 = key.split('-')
                        q1, q2 = data['q1'], data['q2']
                        side = data['side']
                        
                        sig1 = "SELL" if side == 'LONG' else "BUY"
                        sig2 = "BUY" if side == 'LONG' else "SELL"
                        
                        self.executor.execute({"symbol": s1, "signal": sig1, "quantity": q1, "price": 0, "strategy": "TIME_EXIT"})
                        self.executor.execute({"symbol": s2, "signal": sig2, "quantity": q2, "price": 0, "strategy": "TIME_EXIT"})
                    else: # MOMENTUM
                        self.close_position(key, 0, "TIME_EXIT")
                self.open_positions.clear()
                sys.exit(0)

    def start(self):
        print("\n🟢 Engine is Running... (Press Ctrl+C to Stop)")
        try:
            while True:
                now = datetime.now()
                if now.second == 0: 
                    print(f"\n⏰ Tick: {now.strftime('%H:%M:%S')}")
                    self.check_global_exits()
                    self.run_momentum_strategy()
                    self.run_pairs_strategy()
                    self.print_portfolio_status()
                    time.sleep(55) 
                else: time.sleep(1)
        except KeyboardInterrupt: print("\n🔴 Engine Stopped.")
