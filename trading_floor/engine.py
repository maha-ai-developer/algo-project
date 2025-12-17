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
# Set these to False to disable a specific strategy
RUN_MOMENTUM = True
RUN_PAIRS = True

class TradingEngine:
    def __init__(self, mode="paper"):
        self.mode = mode
        self.kite = get_kite()
        self.executor = ExecutionHandler(mode=mode)
        
        # Risk System (Capital: 15k, Max Risk: 2%)
        self.risk_manager = RiskManager(total_capital=15000, max_risk_pct=0.02)

        self.momentum_config = self.load_json(config.MOMENTUM_CONFIG)
        self.pairs_config = self.load_json(config.PAIRS_CONFIG)
        
        # State Management
        self.last_processed_candle = {} 
        self.open_positions = {} # Tracks Symbols (e.g., 'INFY') AND Pairs (e.g., 'INFY-TCS')
        self.active_risk_state = {} 

        print(f"\n✨ ENGINE STARTED in {mode.upper()} mode")
        print(f"   🛡️ SAFETY LOCK: Qty forced to 1 per trade.")
        print(f"   🛡️ SAFETY LOCK: Duplicate entry prevention active.")

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
            # Handle Pairs Display vs Momentum Display
            if '-' in key: # Pair
                status_data.append({"Asset": key, "Type": "PAIR", "Side": side, "Status": "Active"})
                continue

            # Momentum
            risk = self.active_risk_state.get(key, {})
            entry = risk.get('entry', 0)
            sl = risk.get('sl', 0)
            
            df = self.fetch_latest_candles(key, "5minute")
            ltp = df['close'].iloc[-1] if df is not None else entry
            pnl = (ltp - entry) if side == 'LONG' else (entry - ltp)
            
            status_data.append({
                "Asset": key, "Type": "MOMENTUM", "Side": side,
                "Entry": round(entry, 2), "LTP": round(ltp, 2),
                "SL": round(sl, 2), "PnL": round(pnl, 2)
            })
            
        print("\n" + tabulate(status_data, headers="keys", tablefmt="simple_grid"))

    # =================================================================
    # MOMENTUM STRATEGY
    # =================================================================
    def run_momentum_strategy(self):
        if not RUN_MOMENTUM: return

        for symbol, params in self.momentum_config.items():
            time.sleep(0.2)
            
            # 1. DUPLICATE CHECK: Skip if already in position
            if symbol in self.open_positions:
                # We still process it ONLY for Exit Logic, not new Entry
                pass 
            else:
                # If we are checking for entry, ensure we aren't holding it
                pass 

            df = self.fetch_latest_candles(symbol, "5minute")
            if df is None or df.empty: continue
            
            # Stale Check
            last_candle_time = df.index[-1]
            if last_candle_time.tzinfo is not None: last_candle_time = last_candle_time.tz_localize(None)
            if (datetime.now() - last_candle_time).total_seconds() > 900: continue
            
            if self.last_processed_candle.get(symbol) == last_candle_time: continue
            self.last_processed_candle[symbol] = last_candle_time

            # Indicators
            ema_len = params.get('ema_period', 50)
            df['ema'] = ta.ema(df['close'], length=ema_len)
            df['rsi'] = ta.rsi(df['close'], length=14)
            atr = self.risk_manager.calculate_atr(df)
            current = df.iloc[-1]
            price = current['close']
            
            # --- EXIT LOGIC ---
            if symbol in self.open_positions:
                risk_state = self.active_risk_state.get(symbol, {})
                current_sl = risk_state.get('sl', 0)
                highest = risk_state.get('highest_price', price)
                
                # Chandelier Update
                if price > highest: self.active_risk_state[symbol]['highest_price'] = price
                new_sl = self.risk_manager.calculate_chandelier_exit(
                    price, current_sl, self.active_risk_state[symbol]['highest_price'], atr, "LONG"
                )
                if new_sl > current_sl:
                    self.active_risk_state[symbol]['sl'] = new_sl
                    print(f"   🛡️ {symbol} TSL Up -> {new_sl:.2f}")
                    current_sl = new_sl

                if price < current_sl: self.close_position(symbol, price, "SL_HIT")
                elif current['rsi'] < params.get('rsi_exit', 40) or price < current['ema']:
                    self.close_position(symbol, price, "SIGNAL_EXIT")

            # --- ENTRY LOGIC ---
            elif symbol not in self.open_positions:
                rsi_entry = params.get('rsi_entry', 60)
                if current['rsi'] > rsi_entry and price > current['ema']:
                    
                    # Risk Calc (Just for SL distance, Quantity is forced)
                    _, sl_price, dist = self.risk_manager.size_position(symbol, price, atr)
                    
                    # 🔒 SAFETY OVERRIDE: FORCE QUANTITY 1
                    qty = 1 
                    
                    self.executor.execute({
                        "symbol": symbol, "signal": "BUY", "quantity": qty,
                        "price": price, "strategy": "momentum"
                    })
                    self.open_positions[symbol] = 'LONG'
                    self.active_risk_state[symbol] = {
                        'entry': price, 'sl': sl_price, 'atr': atr, 'highest_price': price
                    }
                    print(f"   🛡️ SL Set @ {sl_price:.2f} (Qty Locked: 1)")

    # =================================================================
    # PAIRS STRATEGY
    # =================================================================
    def run_pairs_strategy(self):
        if not RUN_PAIRS: return

        for pair_cfg in self.pairs_config:
            s1 = pair_cfg['leg1']
            s2 = pair_cfg['leg2']
            pair_key = f"{s1}-{s2}"
            
            # 🔒 SAFETY: Check conflicts with Momentum
            if s1 in self.open_positions or s2 in self.open_positions:
                # If either stock is busy in Momentum, skip this Pair
                continue
            
            time.sleep(0.2)
            df1 = self.fetch_latest_candles(s1, "5minute")
            df2 = self.fetch_latest_candles(s2, "5minute")
            if df1 is None or df2 is None: continue
            
            df = pd.concat([df1['close'], df2['close']], axis=1).dropna()
            df.columns = ['l1', 'l2']
            if df.empty: continue
            
            # Stale Check (using s1 index)
            last_t = df.index[-1]
            if last_t.tzinfo: last_t = last_t.tz_localize(None)
            if (datetime.now() - last_t).total_seconds() > 900: continue

            ratio = df['l1'] / df['l2']
            zscore = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std()
            curr_z = zscore.iloc[-1]
            
            entry_z = pair_cfg.get('entry_z', 2.0)
            exit_z = pair_cfg.get('exit_z', 0.5)

            # ENTRY
            if pair_key not in self.open_positions:
                # 🔒 SAFETY OVERRIDE: QTY 1
                qty = 1
                
                if curr_z > entry_z: # Short Spread
                    print(f"   ⚡ PAIR ENTER: {pair_key} (Short)")
                    self.executor.execute({"symbol": s1, "signal": "SELL", "quantity": qty, "price": df['l1'].iloc[-1], "strategy": "pair"})
                    self.executor.execute({"symbol": s2, "signal": "BUY", "quantity": qty, "price": df['l2'].iloc[-1], "strategy": "pair"})
                    self.open_positions[pair_key] = 'SHORT'
                
                elif curr_z < -entry_z: # Long Spread
                    print(f"   ⚡ PAIR ENTER: {pair_key} (Long)")
                    self.executor.execute({"symbol": s1, "signal": "BUY", "quantity": qty, "price": df['l1'].iloc[-1], "strategy": "pair"})
                    self.executor.execute({"symbol": s2, "signal": "SELL", "quantity": qty, "price": df['l2'].iloc[-1], "strategy": "pair"})
                    self.open_positions[pair_key] = 'LONG'

            # EXIT
            elif pair_key in self.open_positions:
                state = self.open_positions[pair_key]
                qty = 1
                if abs(curr_z) < exit_z:
                    print(f"   ⚡ PAIR EXIT: {pair_key} (Mean Reversion)")
                    sig1 = "BUY" if state == 'SHORT' else "SELL"
                    sig2 = "SELL" if state == 'SHORT' else "BUY"
                    self.executor.execute({"symbol": s1, "signal": sig1, "quantity": qty, "price": df['l1'].iloc[-1], "strategy": "pair_exit"})
                    self.executor.execute({"symbol": s2, "signal": sig2, "quantity": qty, "price": df['l2'].iloc[-1], "strategy": "pair_exit"})
                    del self.open_positions[pair_key]

    def close_position(self, symbol, price, reason):
        if symbol in self.open_positions:
            self.executor.execute({
                "symbol": symbol, "signal": "SELL", "quantity": 1,
                "price": price, "strategy": reason
            })
            print(f"   🛑 {symbol} Closed: {reason}")
            del self.open_positions[symbol]
            if symbol in self.active_risk_state: del self.active_risk_state[symbol]

    def check_global_exits(self):
        if self.risk_manager.check_kill_switch(): sys.exit(0)
        if self.risk_manager.check_time_exit():
            if self.open_positions:
                print("   ⏰ AUTO SQUARE OFF")
                # Close Momentum
                for k in list(self.open_positions.keys()):
                    if '-' not in k: self.close_position(k, 0, "TIME_EXIT")
                # Close Pairs (Simplified: Just dumping dict for now in demo)
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
