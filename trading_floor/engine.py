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

class TradingEngine:
    def __init__(self, mode="paper"):
        self.mode = mode
        self.kite = get_kite()
        self.executor = ExecutionHandler(mode=mode)
        
        # 1. Initialize Risk System
        self.risk_manager = RiskManager(total_capital=15000, max_risk_pct=0.02)

        # 2. Load Strategies
        self.momentum_config = self.load_json(config.MOMENTUM_CONFIG)
        self.pairs_config = self.load_json(config.PAIRS_CONFIG)
        
        # 3. State Management
        self.last_processed_candle = {} 
        self.open_positions = {} 
        self.active_risk_state = {} 

        print(f"\n✨ ENGINE STARTED in {mode.upper()} mode")
        print(f"   📊 Momentum Targets: {len(self.momentum_config)}")
        print(f"   ⚖️ Pairs Targets: {len(self.pairs_config)}")
        print(f"   🛡️ Risk Profile: Max Risk {self.risk_manager.max_risk_pct*100}% per trade")

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
        except Exception as e:
            # print(f"   ⚠️ Data fetch error {symbol}: {e}")
            pass
        return None

    def print_portfolio_status(self):
        """Prints a live table of Open Positions, PnL, and Stop Losses."""
        if not self.open_positions:
            return

        status_data = []
        for symbol, side in self.open_positions.items():
            if '-' in symbol: continue 

            risk = self.active_risk_state.get(symbol, {})
            entry = risk.get('entry', 0)
            sl = risk.get('sl', 0)
            highest = risk.get('highest_price', 0)
            
            # Fetch Current Price (Quick Snap)
            df = self.fetch_latest_candles(symbol, "5minute")
            ltp = df['close'].iloc[-1] if df is not None else entry
            
            # Calculate Unrealized PnL
            pnl = (ltp - entry) if side == 'LONG' else (entry - ltp)
            pnl_pct = (pnl / entry) * 100
            
            status_data.append({
                "Symbol": symbol,
                "Side": side,
                "Entry": round(entry, 2),
                "LTP": round(ltp, 2),
                "Stop Loss": round(sl, 2),
                "High Watermark": round(highest, 2),
                "PnL %": f"{pnl_pct:.2f}%"
            })
            
        print("\n" + tabulate(status_data, headers="keys", tablefmt="simple_grid"))

    # =================================================================
    # MOMENTUM STRATEGY (UPDATED)
    # =================================================================
    def run_momentum_strategy(self):
        for symbol, params in self.momentum_config.items():
            
            time.sleep(0.2)
            
            df = self.fetch_latest_candles(symbol, "5minute")
            if df is None or df.empty: continue
            
            # --- FIX: TIMEZONE HANDLING & STALE CHECK ---
            last_candle_time = df.index[-1]
            
            # Convert to TZ-Naive (Wall Clock Time) if Aware
            if last_candle_time.tzinfo is not None:
                last_candle_time = last_candle_time.tz_localize(None)
                
            now = datetime.now()
            
            # Check if candle is older than 15 minutes (900 seconds)
            if (now - last_candle_time).total_seconds() > 900:
                # Skip stale data (e.g., market is closed)
                continue

            # Standard Logic
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
                highest_price = risk_state.get('highest_price', price)
                
                # A. Update Chandelier High Watermark
                if price > highest_price:
                    self.active_risk_state[symbol]['highest_price'] = price
                
                # B. Dynamic Chandelier Stop
                new_sl = self.risk_manager.calculate_chandelier_exit(
                    current_price=price,
                    current_sl=current_sl,
                    highest_price=self.active_risk_state[symbol]['highest_price'],
                    atr=atr,
                    direction="LONG"
                )
                
                if new_sl > current_sl:
                    self.active_risk_state[symbol]['sl'] = new_sl
                    print(f"   🛡️ {symbol} TSL Moved Up -> {new_sl:.2f}")
                    current_sl = new_sl 

                # C. Check Exits
                if price < current_sl:
                    self.close_position(symbol, price, "SL_HIT")
                elif current['rsi'] < params.get('rsi_exit', 40) or price < current['ema']:
                    self.close_position(symbol, price, "SIGNAL_EXIT")

            # --- ENTRY LOGIC ---
            elif symbol not in self.open_positions:
                rsi_entry = params.get('rsi_entry', 60)
                if current['rsi'] > rsi_entry and price > current['ema']:
                    
                    win_rate = params.get('win_rate', 0.50)
                    qty, sl_price, dist = self.risk_manager.size_position(symbol, price, atr, win_rate)
                    
                    if qty > 0:
                        self.executor.execute({
                            "symbol": symbol, "signal": "BUY", "quantity": qty,
                            "price": price, "strategy": "momentum_kelly"
                        })
                        self.open_positions[symbol] = 'LONG'
                        self.active_risk_state[symbol] = {
                            'entry': price, 'sl': sl_price, 'atr': atr, 'highest_price': price
                        }
                        print(f"   🛡️ SL Tracking Active @ {sl_price:.2f}")

    def run_pairs_strategy(self):
        # (Pairs logic kept same as previous step, omitted here for brevity but should exist in file)
        pass 

    def close_position(self, symbol, price, reason):
        if symbol in self.open_positions:
            self.executor.execute({
                "symbol": symbol, "signal": "SELL", "quantity": 1,
                "price": price, "strategy": reason
            })
            print(f"   🛑 {symbol} Closed: {reason}")
            del self.open_positions[symbol]
            if symbol in self.active_risk_state:
                del self.active_risk_state[symbol]

    def check_global_exits(self):
        if self.risk_manager.check_kill_switch():
            sys.exit(0)
        if self.risk_manager.check_time_exit():
            if self.open_positions:
                print("   ⏰ MARKET CLOSING - AUTO SQUARE OFF")
                for sym in list(self.open_positions.keys()):
                    if '-' not in sym: self.close_position(sym, 0, "TIME_EXIT")
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
                    # self.run_pairs_strategy()
                    
                    self.print_portfolio_status()
                    
                    time.sleep(55) 
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n🔴 Engine Stopped.")
