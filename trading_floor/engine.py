"""
Trading Engine v2.0 - Optimized Architecture

Implements all 6 optimizations:
#1: Parallel API Calls (ThreadPoolExecutor via DataCache)
#2: Incremental Updates (DataCache)
#3: Dependency Injection (loose coupling)
#4: State Persistence (StateManager)
#5: Guardian Caching (in guardian.py)
#6: WebSocket Real-Time (RealtimeTicker)
"""

import time
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infrastructure.config as config
from strategies.pairs import PairStrategy

# --- ⚙️ ENGINE CONFIGURATION ---
# "NRML" = Futures Overnight (Required per Futures Trading.pdf)
PRODUCT_TYPE = "NRML"

# Capital per pair
TOTAL_CAPITAL = 50000

# Processing settings
MAX_PARALLEL_WORKERS = 5
PROCESS_INTERVAL_SEC = 60


class TradingEngine:
    """
    Optimized Trading Engine with Dependency Injection.
    
    Improvements over v1:
    - Accepts dependencies via constructor (testable, loosely coupled)
    - Uses DataCache for parallel/incremental data fetching
    - Persists state for crash recovery
    - Optional WebSocket for real-time updates
    """
    
    def __init__(
        self,
        broker,
        data_cache,
        state_manager,
        executor_handler,
        risk_manager,
        ticker=None,
        mode: str = "paper"
    ):
        """
        Dependency Injection constructor.
        
        Args:
            broker: Authenticated Kite client
            data_cache: DataCache instance for price data
            state_manager: StateManager for persistence
            executor_handler: ExecutionHandler for orders
            risk_manager: RiskManager for position sizing
            ticker: Optional RealtimeTicker for WebSocket
            mode: "PAPER" or "LIVE"
        """
        self.mode = mode.upper()
        self.broker = broker
        self.data_cache = data_cache
        self.state_manager = state_manager
        self.executor = executor_handler
        self.risk_manager = risk_manager
        self.ticker = ticker
        
        print(f"\n--- 🚀 STAT ARB ENGINE v2.0 ({self.mode}) ---")
        print(f"   ⚙️ Product Type: {PRODUCT_TYPE}")
        print(f"   💰 Capital Base: ₹{TOTAL_CAPITAL:,}")
        print(f"   ⚡ Parallel Workers: {MAX_PARALLEL_WORKERS}")
        
        # Load persisted state (Optimization #4)
        self.active_trades = self.state_manager.load()
        if self.active_trades:
            print(f"   📂 Restored {len(self.active_trades)} active trades from state")
        
        # Load pair configuration
        self.pairs_config = self._load_pairs_config()
        
        # Cache instrument tokens
        self.tokens = self._load_instrument_tokens()
        self.data_cache.set_tokens(self.tokens)
        
        # Initialize strategy instances
        self.strategies: Dict[str, PairStrategy] = {}
        for p in self.pairs_config:
            pair_key = f"{p['leg1']}-{p['leg2']}"
            print(f"   🧠 Loaded Agent: {pair_key:<20} (Beta: {p['hedge_ratio']:.2f})")
            self.strategies[pair_key] = PairStrategy(
                hedge_ratio=p['hedge_ratio'],
                intercept=p['intercept']
            )
    
    def _load_pairs_config(self) -> list:
        """Load pair configuration from JSON."""
        if not os.path.exists(config.PAIRS_CONFIG):
            print(f"❌ Config not found: {config.PAIRS_CONFIG}")
            sys.exit(1)
        
        with open(config.PAIRS_CONFIG, "r") as f:
            return json.load(f)
    
    def _load_instrument_tokens(self) -> Dict[str, int]:
        """Cache instrument tokens from broker."""
        print("📊 Fetching Instrument Tokens...")
        tokens = {}
        try:
            instruments = self.broker.instruments("NSE")
            inst_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
            
            for p in self.pairs_config:
                if p['leg1'] in inst_map:
                    tokens[p['leg1']] = inst_map[p['leg1']]
                if p['leg2'] in inst_map:
                    tokens[p['leg2']] = inst_map[p['leg2']]
            
            print(f"   ✅ Cached {len(tokens)} tokens")
        except Exception as e:
            print(f"   ⚠️ Failed to fetch instruments: {e}")
        
        return tokens
    
    def run(self):
        """
        Main engine loop.
        
        Uses parallel data fetching and optional WebSocket updates.
        """
        print(f"\n✅ Engine Running... Monitoring {len(self.pairs_config)} Pairs.")
        print("   (Press Ctrl+C to Stop)\n")
        
        # Start WebSocket if available (Optimization #6)
        if self.ticker:
            token_list = list(self.tokens.values())
            self.ticker.connect(token_list, on_price_update=self._on_realtime_tick)
        
        try:
            while True:
                print(f"⏰ Heartbeat: {datetime.now().strftime('%H:%M:%S')}")
                
                # Parallel data fetch for all symbols (Optimization #1)
                self._process_all_pairs_parallel()
                
                # Print cache stats
                stats = self.data_cache.get_stats()
                print(f"   📊 Cache: {stats['cache_hits']} hits, {stats['cache_misses']} misses, {stats['api_calls']} API calls")
                
                time.sleep(PROCESS_INTERVAL_SEC)
                
        except KeyboardInterrupt:
            self._shutdown()
    
    def _process_all_pairs_parallel(self):
        """
        Process all pairs using parallel data fetching.
        """
        # Collect all symbols needed
        all_symbols = set()
        for p in self.pairs_config:
            all_symbols.add(p['leg1'])
            all_symbols.add(p['leg2'])
        
        # Parallel fetch (Optimization #1 + #2)
        price_data = self.data_cache.parallel_fetch(list(all_symbols), interval="day")
        
        # Process each pair with fetched data
        for p in self.pairs_config:
            self._process_pair(p, price_data)
    
    def _process_pair(self, p: dict, price_data: Dict[str, pd.Series]):
        """Process a single pair using pre-fetched data."""
        s1, s2 = p['leg1'], p['leg2']
        pair_key = f"{s1}-{s2}"
        strategy = self.strategies[pair_key]
        
        # Get data from cache
        data_y = price_data.get(s1, pd.Series())
        data_x = price_data.get(s2, pd.Series())
        
        if len(data_y) < 60 or len(data_x) < 60:
            return
        
        # Validate latest prices
        last_y = data_y.iloc[-1]
        last_x = data_x.iloc[-1]
        
        if last_y <= 0 or last_x <= 0 or pd.isna(last_y) or pd.isna(last_x):
            print(f"   🛡️ BLOCKED BAD DATA for {pair_key}")
            return
        
        current_price_y = last_y
        current_price_x = last_x
        
        # Get strategy signal
        response = strategy.generate_signal(data_y, data_x)
        
        health = response.get('health', 'UNKNOWN')
        z = response.get('zscore', 0.0)
        signal = response['signal']
        
        print(f"   👉 {pair_key:<20} | Health: {health:<6} | Z: {z:>5.2f} | Sig: {signal}")
        
        # Risk checks
        if health == "RED":
            if pair_key in self.active_trades:
                print(f"   🔴 GUARDIAN KILL: Closing {pair_key}")
                self._close_position(pair_key, s1, s2, current_price_y, current_price_x)
            return
        
        is_stop, stop_msg = self.risk_manager.check_stop_loss(z, stop_z_threshold=4.0)
        if is_stop and pair_key in self.active_trades:
            print(f"   🟠 STOP LOSS: {stop_msg}")
            self._close_position(pair_key, s1, s2, current_price_y, current_price_x)
            return
        
        is_tp, tp_msg = self.risk_manager.check_take_profit(z)
        if is_tp and pair_key in self.active_trades:
            print(f"   🟢 TAKE PROFIT: {tp_msg}")
            self._close_position(pair_key, s1, s2, current_price_y, current_price_x)
            return
        
        # Entry logic
        if pair_key not in self.active_trades:
            self._handle_entry(pair_key, signal, s1, s2, current_price_y, current_price_x, p['hedge_ratio'])
    
    def _handle_entry(self, pair_key: str, signal: str, s1: str, s2: str, 
                      price_y: float, price_x: float, hedge_ratio: float):
        """Handle trade entry with state persistence."""
        qty_y, qty_x = self.risk_manager.calculate_sizing(price_y, price_x, hedge_ratio)
        
        if qty_y == 0 or qty_x == 0:
            return
        
        if signal == "LONG_SPREAD":
            print(f"   🚀 ENTRY LONG: Buy {qty_y} {s1}, Sell {qty_x} {s2}")
            self.executor.place_pair_order(
                s1, "BUY", qty_y, price_y,
                s2, "SELL", qty_x, price_x,
                product=PRODUCT_TYPE
            )
            trade_data = {"side": "LONG", "q1": qty_y, "q2": qty_x, "entry_time": datetime.now().isoformat()}
            self.active_trades[pair_key] = trade_data
            self.state_manager.save(self.active_trades)  # Persist (Optimization #4)
            
        elif signal == "SHORT_SPREAD":
            print(f"   🚀 ENTRY SHORT: Sell {qty_y} {s1}, Buy {qty_x} {s2}")
            self.executor.place_pair_order(
                s1, "SELL", qty_y, price_y,
                s2, "BUY", qty_x, price_x,
                product=PRODUCT_TYPE
            )
            trade_data = {"side": "SHORT", "q1": qty_y, "q2": qty_x, "entry_time": datetime.now().isoformat()}
            self.active_trades[pair_key] = trade_data
            self.state_manager.save(self.active_trades)  # Persist (Optimization #4)
    
    def _close_position(self, pair_key: str, s1: str, s2: str, px1: float, px2: float):
        """Close position with state persistence."""
        if pair_key not in self.active_trades:
            return
        
        trade = self.active_trades[pair_key]
        print(f"   📉 CLOSING POSITION: {pair_key}")
        
        if trade['side'] == "LONG":
            self.executor.place_pair_order(s1, "SELL", trade['q1'], px1, s2, "BUY", trade['q2'], px2, product=PRODUCT_TYPE)
        else:
            self.executor.place_pair_order(s1, "BUY", trade['q1'], px1, s2, "SELL", trade['q2'], px2, product=PRODUCT_TYPE)
        
        del self.active_trades[pair_key]
        self.state_manager.save(self.active_trades)  # Persist (Optimization #4)
    
    def _on_realtime_tick(self, token: int, price: float, timestamp):
        """Callback for WebSocket price updates (Optimization #6)."""
        # Could trigger immediate re-evaluation for specific pairs
        # For now, just log
        pass
    
    def _shutdown(self):
        """Graceful shutdown."""
        print("\n🛑 Stopping Engine...")
        
        # Save final state
        self.state_manager.save(self.active_trades)
        print(f"   📂 State saved: {len(self.active_trades)} active trades")
        
        # Stop WebSocket
        if self.ticker:
            self.ticker.stop()
        
        print("   ✅ Shutdown complete.")


# ============================================================
# FACTORY FUNCTION (Optimization #3: Dependency Injection)
# ============================================================

def create_engine(mode: str = "paper", use_websocket: bool = False) -> TradingEngine:
    """
    Factory function to create a fully configured TradingEngine.
    
    This provides a clean interface while hiding dependency wiring.
    
    Args:
        mode: "PAPER" or "LIVE"
        use_websocket: Enable WebSocket for real-time updates
        
    Returns:
        Configured TradingEngine instance
    """
    from infrastructure.broker.kite_auth import get_kite
    from infrastructure.data.cache import DataCache
    from trading_floor.state import StateManager
    from trading_floor.execution import ExecutionHandler
    from trading_floor.risk_manager import RiskManager
    
    # Create dependencies
    broker = get_kite()
    cache = DataCache(broker, max_workers=MAX_PARALLEL_WORKERS)
    state = StateManager()
    executor = ExecutionHandler(mode=mode)
    risk = RiskManager(capital_per_pair=TOTAL_CAPITAL)
    
    # Optional WebSocket
    ticker = None
    if use_websocket:
        from infrastructure.broker.ticker import RealtimeTicker
        api_key = config.API_KEY
        access_token = broker.access_token  # Assumes broker has token
        ticker = RealtimeTicker(api_key, access_token)
    
    return TradingEngine(
        broker=broker,
        data_cache=cache,
        state_manager=state,
        executor_handler=executor,
        risk_manager=risk,
        ticker=ticker,
        mode=mode
    )


if __name__ == "__main__":
    # Example usage
    # engine = create_engine(mode="PAPER", use_websocket=False)
    # engine.run()
    pass
