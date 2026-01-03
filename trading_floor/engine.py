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
TOTAL_CAPITAL = 500000

# Processing settings
MAX_PARALLEL_WORKERS = 5
PROCESS_INTERVAL_SEC = 60
MAX_HOLDING_SESSIONS = 25  # Checklist Phase 8: Time-based exit at 25 trading sessions

# === RISK MANAGEMENT (Checklist Phase 9) ===
MAX_CONCURRENT_PAIRS = 5       # Maximum simultaneous pair positions
MAX_LOSS_PER_PAIR = 10000      # Maximum loss per pair (₹10K = 20% of 50K capital)
MAX_PAIRS_PER_SECTOR = 2       # Sector diversification limit

# === PDF-COMPLIANT THRESHOLDS (Zerodha Varsity) ===
Z_ENTRY_THRESHOLD = 2.5        # Entry at ±2.5 SD (Page 47)
Z_EXIT_THRESHOLD = 1.0         # Exit at ±1.0 SD (mean reversion target)
Z_STOP_THRESHOLD = 3.0         # Stop loss at ±3.0 SD
MAX_INTERCEPT_RISK_PCT = 0.20  # Intercept explains max 20% of Y price (Chapter 14)

# === ANTI-CHURNING PROTECTION ===
POST_EXIT_COOLDOWN_HOURS = 24  # Prevent re-entry to same pair for 24 hours after exit

# === DISPLAY SETTINGS ===
VERBOSE = False  # Set to True for detailed per-pair logging
COMPACT_MODE = True  # Show compact table summary instead of scrolling output


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
        
        # Anti-churning: Track when pairs were last exited
        self.exit_cooldowns: Dict[str, datetime] = {}
        
        # Load pair configuration
        self.pairs_config = self._load_pairs_config()
        
        # Detect orphan positions (trades not in current config)
        self.orphan_pairs = self._detect_orphan_positions()
        
        # Cache instrument tokens (include orphan symbols)
        self.tokens = self._load_instrument_tokens()
        self.data_cache.set_tokens(self.tokens)
        
        # Initialize strategy instances
        self.strategies: Dict[str, PairStrategy] = {}
        for p in self.pairs_config:
            # Support both key formats: 'stock_y'/'stock_x' (old) and 'leg1'/'leg2' (new)
            s_y = p.get('stock_y') or p.get('leg1')
            s_x = p.get('stock_x') or p.get('leg2')
            beta = p.get('beta') or p.get('hedge_ratio', 1.0)
            
            pair_key = f"{s_y}-{s_x}"
            print(f"   🧠 Loaded Agent: {pair_key:<20} (Beta: {beta:.2f})")
            self.strategies[pair_key] = PairStrategy(
                hedge_ratio=beta,
                intercept=p.get('intercept', 0.0)
            )
        
        # Create temporary strategies for orphan positions (so they can exit)
        for pair_key, orphan_data in self.orphan_pairs.items():
            print(f"   👻 Orphan Trade: {pair_key:<20} (monitoring for exit)")
            self.strategies[pair_key] = PairStrategy(
                hedge_ratio=orphan_data.get('hedge_ratio', 1.0),
                intercept=orphan_data.get('intercept', 0.0)
            )
    
    def _load_pairs_config(self) -> list:
        """Load pair configuration from JSON."""
        if not os.path.exists(config.PAIRS_CONFIG):
            print(f"❌ Config not found: {config.PAIRS_CONFIG}")
            sys.exit(1)
        
        with open(config.PAIRS_CONFIG, "r") as f:
            return json.load(f)
    
    def _detect_orphan_positions(self) -> Dict[str, Any]:
        """
        Detect orphan positions: active trades for pairs not in current config.
        
        Returns dict of orphan pair_key -> trade data with hedge_ratio fallback.
        """
        if not self.active_trades:
            return {}
        
        # Get all pair keys from current config
        config_pairs = set()
        for p in self.pairs_config:
            s_y = p.get('stock_y') or p.get('leg1')
            s_x = p.get('stock_x') or p.get('leg2')
            config_pairs.add(f"{s_y}-{s_x}")
        
        orphans = {}
        
        for pair_key, trade_data in self.active_trades.items():
            if pair_key not in config_pairs:
                # Try to find hedge_ratio from candidates file
                hedge_ratio = 1.0
                intercept = 0.0
                
                if os.path.exists(config.PAIRS_CANDIDATES_FILE):
                    try:
                        with open(config.PAIRS_CANDIDATES_FILE, 'r') as f:
                            candidates = json.load(f)
                        for c in candidates:
                            c_y = c.get('stock_y') or c.get('leg1')
                            c_x = c.get('stock_x') or c.get('leg2')
                            if f"{c_y}-{c_x}" == pair_key:
                                hedge_ratio = c.get('hedge_ratio') or c.get('beta', 1.0)
                                intercept = c.get('intercept', 0.0)
                                break
                    except:
                        pass
                
                orphans[pair_key] = {
                    **trade_data,
                    'hedge_ratio': hedge_ratio,
                    'intercept': intercept
                }
                print(f"   ⚠️ ORPHAN DETECTED: {pair_key} (not in current config, will monitor for exit)")
        
        return orphans
    
    def _load_instrument_tokens(self) -> Dict[str, int]:
        """Cache instrument tokens from broker."""
        print("📊 Fetching Instrument Tokens...")
        tokens = {}
        try:
            instruments = self.broker.instruments("NSE")
            inst_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
            
            for p in self.pairs_config:
                s_y = p.get('stock_y') or p.get('leg1')
                s_x = p.get('stock_x') or p.get('leg2')
                if s_y in inst_map:
                    tokens[s_y] = inst_map[s_y]
                if s_x in inst_map:
                    tokens[s_x] = inst_map[s_x]
            
            print(f"   ✅ Cached {len(tokens)} tokens")
        except Exception as e:
            print(f"   ⚠️ Failed to fetch instruments: {e}")
        
        # Also add tokens for orphan pair symbols
        for pair_key in self.orphan_pairs:
            s1, s2 = pair_key.split('-')
            if s1 in inst_map:
                tokens[s1] = inst_map[s1]
            if s2 in inst_map:
                tokens[s2] = inst_map[s2]
        
        return tokens
    
    def run(self):
        """
        Main engine loop.
        
        Uses parallel data fetching and optional WebSocket updates.
        """
        print(f"\n✅ Engine Running... Monitoring {len(self.pairs_config)} Pairs.")
        if self.orphan_pairs:
            print(f"   👻 Also tracking {len(self.orphan_pairs)} orphan position(s) for exit.")
        print("   (Press Ctrl+C to Stop)\n")
        
        # Initialize pair results for compact display
        self.pair_results = {}
        
        # Start WebSocket if available (Optimization #6)
        if self.ticker:
            token_list = list(self.tokens.values())
            self.ticker.connect(token_list, on_price_update=self._on_realtime_tick)
        
        try:
            while True:
                # Clear screen for compact mode
                if COMPACT_MODE:
                    print("\033[H\033[J", end="")
                    self._display_compact_header()
                else:
                    print(f"⏰ Heartbeat: {datetime.now().strftime('%H:%M:%S')}")
                
                # Parallel data fetch for all symbols (Optimization #1)
                self._process_all_pairs_parallel()
                
                # Display compact table
                if COMPACT_MODE:
                    self._display_compact_table()
                
                time.sleep(PROCESS_INTERVAL_SEC)
                
        except KeyboardInterrupt:
            self._shutdown()
    
    def _display_compact_header(self):
        """Display compact header for table mode."""
        now = datetime.now().strftime('%d %b %Y  %H:%M:%S')
        mode_str = f"🔴 LIVE" if self.mode == "LIVE" else "🟡 PAPER"
        print("="*90)
        print(f"{'🚀 STAT ARB ENGINE v2.0':^90}")
        print(f"{'⏰ ' + now + ' | ' + mode_str:^90}")
        print("="*90)
        print(f"Active: {len(self.active_trades)} | Monitoring: {len(self.pairs_config)} pairs | Refresh: {PROCESS_INTERVAL_SEC}s")
        print("="*90)
    
    def _display_compact_table(self):
        """Display all pairs in a compact table."""
        # Header
        print()
        print(f"{'PAIR':<24}│{'Z-SCORE':^10}│{'SIGNAL':^12}│{'STATUS':^28}│{'INTERCEPT':^10}")
        print("─"*24 + "┼" + "─"*10 + "┼" + "─"*12 + "┼" + "─"*28 + "┼" + "─"*10)
        
        for pair_key, result in self.pair_results.items():
            z = result.get('z', 0)
            signal = result.get('signal', 'WAIT')
            status = result.get('status', '')
            intercept_pct = result.get('intercept_pct', 0)
            
            # Z-score color indicator
            if abs(z) > 3.0:
                z_str = f"⚠️{z:+.2f}"  # Stop zone
            elif abs(z) > 2.5:
                z_str = f"🔴{z:+.2f}"  # Entry zone
            elif abs(z) < 1.0:
                z_str = f"🎯{z:+.2f}"  # Exit zone
            else:
                z_str = f"{z:+.2f}"
            
            # Signal indicator
            if 'SHORT' in signal:
                sig_str = "📉 SHORT"
            elif 'LONG' in signal:
                sig_str = "📈 LONG"
            elif 'EXIT' in signal:
                sig_str = "🎯 EXIT"
            else:
                sig_str = "⏸️ WAIT"
            
            # Intercept indicator
            if intercept_pct > 20:
                int_str = f"❌{intercept_pct:.0f}%"
            elif intercept_pct > 10:
                int_str = f"⚠️{intercept_pct:.0f}%"
            else:
                int_str = f"✅{intercept_pct:.0f}%"
            
            print(f"{pair_key:<24}│{z_str:^10}│{sig_str:^12}│{status:<28}│{int_str:^10}")
        
        # Footer with active trades
        print("─"*90)
        if self.active_trades:
            print(f"\n📍 ACTIVE POSITIONS: {len(self.active_trades)}")
            for pk, trade in self.active_trades.items():
                entry_z = trade.get('entry_zscore', 0)
                side = trade.get('side', '')
                print(f"   • {pk}: {side} @ Z={entry_z:.2f}")
        else:
            print(f"\n📭 No active positions")
        
        print(f"\n💡 Rules: Entry ±2.5 | Exit ±1.0 | Stop ±3.0 | Cooldown 24h")
    
    def _process_all_pairs_parallel(self):
        """
        Process all pairs using parallel data fetching.
        """
        # Collect all symbols needed (including orphans)
        all_symbols = set()
        for p in self.pairs_config:
            s_y = p.get('stock_y') or p.get('leg1')
            s_x = p.get('stock_x') or p.get('leg2')
            all_symbols.add(s_y)
            all_symbols.add(s_x)
        
        # Add orphan symbols
        for pair_key in self.orphan_pairs:
            s1, s2 = pair_key.split('-')
            all_symbols.add(s1)
            all_symbols.add(s2)
        
        # Parallel fetch (Optimization #1 + #2)
        # Use LIVE fetch (historical + current LTP) for real-time Z-scores
        # Get expiry from config (use first pair's expiry, or None for auto-detect)
        expiry_str = self.pairs_config[0].get('expiry') if self.pairs_config else None
        price_data = self.data_cache.parallel_fetch_live(list(all_symbols), interval="day", expiry_str=expiry_str)
        
        # Process each config pair with fetched data
        for p in self.pairs_config:
            self._process_pair(p, price_data)
        
        # Process orphan pairs (exit-only monitoring)
        # Use list() to copy dict items to avoid RuntimeError when dict changes during iteration
        for pair_key, orphan_data in list(self.orphan_pairs.items()):
            self._process_orphan_pair(pair_key, orphan_data, price_data)
    
    def _process_pair(self, p: dict, price_data: Dict[str, pd.Series]):
        """Process a single pair using pre-fetched data."""
        # Support both key formats
        s1 = p.get('stock_y') or p.get('leg1')
        s2 = p.get('stock_x') or p.get('leg2')
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
        
        # Get config statistics for enhanced output
        adf_pvalue = p.get('adf_pvalue', 0.05)
        std_err = p.get('std_err', 0)
        intercept = p.get('intercept', 0)
        beta = p.get('beta', 1.0)
        lot_y = p.get('lot_size_y', 1)
        lot_x = p.get('lot_size_x', 1)
        
        # Intercept Ratio Check (for model validity)
        intercept_ratio = (abs(intercept) / current_price_y * 100) if current_price_y > 0 else 0
        
        if intercept_ratio > MAX_INTERCEPT_RISK_PCT * 100:
            model_valid = False
        else:
            model_valid = True
        
        # Determine signal and status for compact display
        if abs(z) > Z_STOP_THRESHOLD:
            sig_str = "BLOCKED"
            status_str = f"Z past stop ({Z_STOP_THRESHOLD})"
        elif abs(z) > Z_ENTRY_THRESHOLD:
            sig_str = "SHORT" if z > 0 else "LONG"
            status_str = f"Entry signal ({'valid' if model_valid else 'invalid'})"
        elif abs(z) < Z_EXIT_THRESHOLD:
            sig_str = "EXIT"
            status_str = "Mean reversion zone"
        else:
            sig_str = "WAIT"
            status_str = f"{Z_EXIT_THRESHOLD}<|Z|<{Z_ENTRY_THRESHOLD}"
        
        # Store result for compact table
        self.pair_results[pair_key] = {
            'z': z,
            'signal': sig_str,
            'status': status_str,
            'intercept_pct': intercept_ratio,
            'health': health,
            'model_valid': model_valid
        }
        
        # Verbose logging (only when VERBOSE=True)
        if VERBOSE:
            print(f"   {'─'*60}")
            print(f"   👉 {pair_key} | {health} | Z={z:+.2f}")
            stationarity_pct = (1 - adf_pvalue) * 100
            adf_pass = adf_pvalue < 0.05
            print(f"   📊 ADF: p={adf_pvalue:.3f} → {stationarity_pct:.1f}% stationary {'✓' if adf_pass else '✗'}")
            if not model_valid:
                print(f"   🚫 INTERCEPT: {intercept_ratio:.0f}% > {MAX_INTERCEPT_RISK_PCT*100:.0f}% → REJECT")
            elif intercept_ratio > 10:
                print(f"   ⚠️ INTERCEPT: {intercept_ratio:.0f}% → MARGINAL")
            else:
                print(f"   ✅ INTERCEPT: {intercept_ratio:.0f}% → EXCELLENT")
        
        # 3. Beta-Neutral Position Sizing
        if lot_x >= lot_y:
            required_y = lot_x / beta
            lots_y_calc = max(1, round(required_y / lot_y))
            lots_x_calc = 1
        else:
            required_x = lot_y * beta
            lots_x_calc = max(1, round(required_x / lot_x))
            lots_y_calc = 1
        
        shares_y = lots_y_calc * lot_y
        shares_x = lots_x_calc * lot_x
        beta_required_x = shares_y * beta
        
        # Calculate mismatch
        mismatch_shares = shares_x - beta_required_x
        mismatch_pct = abs(mismatch_shares) / beta_required_x * 100 if beta_required_x > 0 else 0
        
        if VERBOSE:
            print(f"   💼 SIZING: {lots_y_calc}L×{lot_y}={shares_y} Y | {lots_x_calc}L×{lot_x}={shares_x} X")
            print(f"      ⚖️ Beta-Neutral: {shares_y}×{beta:.2f}={beta_required_x:.0f} vs {shares_x}")
        
        # 3a. Try to find better scaling combination (up to 5x)
        best_lots_y, best_lots_x, best_mismatch = lots_y_calc, lots_x_calc, mismatch_pct
        for scale in range(2, 6):
            scaled_y = lots_y_calc * scale
            scaled_x = lots_x_calc * scale
            scaled_shares_y = scaled_y * lot_y
            scaled_shares_x = scaled_x * lot_x
            scaled_required = scaled_shares_y * beta
            scaled_mismatch = abs(scaled_shares_x - scaled_required) / scaled_required * 100 if scaled_required > 0 else 0
            if scaled_mismatch < best_mismatch:
                best_lots_y, best_lots_x, best_mismatch = scaled_y, scaled_x, scaled_mismatch
        
        # 3b. Mismatch Decision
        if mismatch_pct > 20:
            if best_mismatch < mismatch_pct * 0.7:
                if VERBOSE: print(f"      ⚠️ MISMATCH: {mismatch_pct:.1f}% → TRY {best_lots_y}L×{best_lots_x}L = {best_mismatch:.1f}%")
            else:
                if VERBOSE: print(f"      🚫 MISMATCH: {mismatch_pct:.1f}% > 20% threshold")
                model_valid = False
        elif mismatch_pct > 10:
            if VERBOSE: print(f"      ⚠️ MISMATCH: {mismatch_pct:.1f}% → Consider spot adjustment")
        else:
            if VERBOSE: print(f"      ✅ MISMATCH: {mismatch_pct:.1f}% → ACCEPTABLE")
        
        # 4. Z-Score Signal (PDF-Compliant: Entry ±2.5, Exit ±1.0)
        if VERBOSE:
            if abs(z) > 2.5:
                if z > 2.5:
                    direction = f"SHORT {s1} + LONG {s2}"
                else:
                    direction = f"LONG {s1} + SHORT {s2}"
                print(f"   📈 Z={z:+.2f} → ENTRY: {direction}")
            elif abs(z) < 1.0:
                print(f"   📈 Z={z:+.2f} → EXIT zone (mean reversion)")
            else:
                print(f"   📈 Z={z:+.2f} → WAIT zone")
        
        # 5. DECISION RULE
        if not model_valid:
            if VERBOSE: print(f"   ❌ DECISION: REJECT - Model invalid or mismatch >20%")
            return  # Block this pair from trading
        
        # Risk checks
        if health == "RED":
            if pair_key in self.active_trades:
                print(f"   🔴 GUARDIAN KILL: Closing {pair_key}")
                self._close_position(pair_key, s1, s2, current_price_y, current_price_x, "GUARDIAN_HALT")
            return
        
        is_stop, stop_msg = self.risk_manager.check_stop_loss(z, stop_z_threshold=Z_STOP_THRESHOLD)
        if is_stop and pair_key in self.active_trades:
            print(f"   🟠 STOP LOSS: {stop_msg}")
            self._close_position(pair_key, s1, s2, current_price_y, current_price_x, "STOP_LOSS")
            return
        
        is_tp, tp_msg = self.risk_manager.check_take_profit(z)
        if is_tp and pair_key in self.active_trades:
            print(f"   🟢 TAKE PROFIT: {tp_msg}")
            self._close_position(pair_key, s1, s2, current_price_y, current_price_x, "TAKE_PROFIT")
            return
        
        # Time-based exit check
        if pair_key in self.active_trades:
            trade = self.active_trades[pair_key]
            entry_time = trade.get('entry_time')
            if entry_time:
                entry_date = datetime.fromisoformat(entry_time).date()
                days_held = (datetime.now().date() - entry_date).days
                sessions_held = int(days_held * 5 / 7)
                if sessions_held >= MAX_HOLDING_SESSIONS:
                    print(f"   ⏰ TIME EXIT: {pair_key} held {sessions_held} sessions")
                    self._close_position(pair_key, s1, s2, current_price_y, current_price_x, "TIME_EXIT")
                    return
            
            # Position Monitoring - Calculate Unrealized P&L
            entry_y = trade.get('entry_price_y', current_price_y)
            entry_x = trade.get('entry_price_x', current_price_x)
            qty_y = trade.get('q1', 0)
            qty_x = trade.get('q2', 0)
            side = trade.get('side', 'LONG')
            
            if side == "LONG":
                pnl_y = (current_price_y - entry_y) * qty_y
                pnl_x = (entry_x - current_price_x) * qty_x
            else:
                pnl_y = (entry_y - current_price_y) * qty_y
                pnl_x = (current_price_x - entry_x) * qty_x
            
            net_pnl = pnl_y + pnl_x
            
            # Update trade_data
            trade['current_zscore'] = z
            trade['unrealized_pnl'] = round(net_pnl, 2)
            trade['last_update'] = datetime.now().isoformat()
            self.state_manager.save(self.active_trades)
            
            # Log position status (always show active positions)
            if VERBOSE:
                pnl_symbol = "📈" if net_pnl > 0 else "📉"
                print(f"   {pnl_symbol} POSITION: {pair_key} | Z: {z:.2f} | P&L: ₹{net_pnl:,.0f}")
        
        # Entry logic
        if pair_key not in self.active_trades:
            if abs(z) > Z_STOP_THRESHOLD:
                if VERBOSE: print(f"   🚫 BLOCKED: Z={z:.2f} past stop loss ({Z_STOP_THRESHOLD})")
                return
            
            # Only attempt entry on valid entry signals (LONG_SPREAD or SHORT_SPREAD)
            if signal not in ["LONG_SPREAD", "SHORT_SPREAD"]:
                return
            
            beta = p.get('beta') or p.get('hedge_ratio', 1.0)
            self._handle_entry(
                pair_key, signal, s1, s2, 
                current_price_y, current_price_x, 
                beta, p.get('intercept', 0),
                entry_zscore=z,  # Phase 6.3: Record entry Z-Score
                sector=p.get('sector', 'UNKNOWN')  # Phase 9: Sector diversification
            )
    

    def _process_orphan_pair(self, pair_key: str, orphan_data: Dict, price_data: Dict[str, pd.Series]):
        """
        Process an orphan pair: exit-only monitoring (no new entries).
        
        Orphan pairs are positions from pairs no longer in the active config.
        We continue monitoring them for mean reversion exit.
        """
        s1, s2 = pair_key.split('-')
        
        if pair_key not in self.strategies:
            return
        
        strategy = self.strategies[pair_key]
        
        # Get data from cache
        data_y = price_data.get(s1, pd.Series())
        data_x = price_data.get(s2, pd.Series())
        
        if len(data_y) < 60 or len(data_x) < 60:
            print(f"   👻 {pair_key:<20} | Insufficient data for orphan monitoring")
            return
        
        # Validate latest prices
        last_y = data_y.iloc[-1]
        last_x = data_x.iloc[-1]
        
        if last_y <= 0 or last_x <= 0 or pd.isna(last_y) or pd.isna(last_x):
            return
        
        current_price_y = last_y
        current_price_x = last_x
        
        # Get strategy signal
        response = strategy.generate_signal(data_y, data_x)
        
        health = response.get('health', 'UNKNOWN')
        z = response.get('zscore', 0.0)
        signal = response['signal']
        
        print(f"   👻 {pair_key:<20} | Health: {health:<6} | Z: {z:>5.2f} | Sig: {signal} (ORPHAN)")
        
        # EXIT LOGIC ONLY (no new entries for orphans)
        if pair_key in self.active_trades:
            # Check for exit conditions
            is_tp, tp_msg = self.risk_manager.check_take_profit(z)
            is_stop, stop_msg = self.risk_manager.check_stop_loss(z, stop_z_threshold=Z_STOP_THRESHOLD)  # PDF: ±3.0
            
            if health == "RED":
                print(f"   👻 ORPHAN GUARDIAN EXIT: {pair_key}")
                self._close_position(pair_key, s1, s2, current_price_y, current_price_x)
                del self.orphan_pairs[pair_key]  # Remove from orphans after exit
                
            elif is_tp:
                print(f"   👻 ORPHAN TAKE PROFIT: {pair_key}")
                self._close_position(pair_key, s1, s2, current_price_y, current_price_x, "TP_ORPHAN")
                del self.orphan_pairs[pair_key]
                
            elif is_stop:
                print(f"   👻 ORPHAN STOP LOSS: {pair_key}")
                self._close_position(pair_key, s1, s2, current_price_y, current_price_x, "SL_ORPHAN")
                del self.orphan_pairs[pair_key]
    
    def _handle_entry(self, pair_key: str, signal: str, s1: str, s2: str, 
                      price_y: float, price_x: float, hedge_ratio: float,
                      intercept: float = 0, entry_zscore: float = 0,
                      sector: str = "UNKNOWN"):
        """
        Handle trade entry with state persistence and full documentation.
        
        Phase 6.3 Trade Documentation:
        - Entry timestamp, Z-Score, prices, sizes
        - Beta, intercept, target/stop Z-Score
        """
        # Pre-Trade Validation (Checklist Phase 4.3)
        
        # 1. Negative beta warning (Checklist Phase 11)
        if hedge_ratio < 0:
            print(f"   ⚠️ WARNING: {pair_key} has NEGATIVE beta ({hedge_ratio:.3f}) - potentially non-tradeable")
        
        # 2. Market hours check (Checklist Phase 4.3)
        if not self._is_market_open():
            print(f"   🚫 BLOCKED: Market closed - no new entries")
            return
        
        # === PHASE 9: RISK MANAGEMENT CHECKS ===
        
        # 3. Maximum concurrent pairs check
        if len(self.active_trades) >= MAX_CONCURRENT_PAIRS:
            print(f"   🚫 BLOCKED: Max pairs ({MAX_CONCURRENT_PAIRS}) reached")
            return
        
        # 4. Sector diversification check
        sector_count = sum(1 for t in self.active_trades.values() if t.get('sector') == sector)
        if sector_count >= MAX_PAIRS_PER_SECTOR:
            print(f"   🚫 BLOCKED: Max pairs per sector ({MAX_PAIRS_PER_SECTOR}) in {sector}")
            return
        
        # 5. Check for existing losses exceeding limit
        for pk, trade in self.active_trades.items():
            unrealized = trade.get('unrealized_pnl', 0)
            if unrealized < -MAX_LOSS_PER_PAIR:
                print(f"   ⚠️ WARNING: {pk} has loss ₹{abs(unrealized):,.0f} > max ₹{MAX_LOSS_PER_PAIR:,}")
        
        # 3. Margin availability check (Checklist Phase 4.1/4.2)
        from trading_floor.risk.sizing import check_margin_availability, get_futures_margin
        
        # Estimate margin for both legs
        estimated_qty_y = max(1, int(TOTAL_CAPITAL / price_y / 10))  # Rough estimate
        estimated_qty_x = max(1, int(abs(hedge_ratio) * estimated_qty_y))
        
        margin_y = get_futures_margin(s1, estimated_qty_y, price_y)
        margin_x = get_futures_margin(s2, estimated_qty_x, price_x)
        total_margin = margin_y + margin_x
        
        is_sufficient, margin_msg = check_margin_availability(
            equity=TOTAL_CAPITAL, 
            initial_margin=total_margin
        )
        
        if not is_sufficient:
            print(f"   🚫 BLOCKED: {margin_msg}")
            return
        
        # 6. Liquidity/Spread Validation (NEW - Checklist Phase 4.2)
        from trading_floor.risk.liquidity import LiquidityChecker
        liquidity_checker = LiquidityChecker()
        
        # In live trading, we'd get real bid/ask from broker
        # For now, estimate spread as 0.1% of price (typical for F&O)
        est_spread_y = price_y * 0.001
        est_spread_x = price_x * 0.001
        bid_y, ask_y = price_y - est_spread_y/2, price_y + est_spread_y/2
        bid_x, ask_x = price_x - est_spread_x/2, price_x + est_spread_x/2
        
        liq_ok, liq_msg, liq_details = liquidity_checker.validate_entry(
            sym_y=s1, bid_y=bid_y, ask_y=ask_y, qty_y=estimated_qty_y,
            sym_x=s2, bid_x=bid_x, ask_x=ask_x, qty_x=estimated_qty_x
        )
        
        if not liq_ok:
            print(f"   🚫 BLOCKED: {liq_msg}")
            return
        # Liquidity OK - proceed silently
        
        # Get lot sizes for proper beta-neutral sizing (Phase 5)
        from infrastructure.data.futures_utils import get_lot_size
        lot_y = get_lot_size(s1)
        lot_x = get_lot_size(s2)
        
        qty_y, qty_x = self.risk_manager.calculate_sizing(
            price_y, price_x, hedge_ratio, lot_y=lot_y, lot_x=lot_x
        )
        
        if qty_y == 0 or qty_x == 0:
            return
        
        # Phase 6.3: Build comprehensive trade documentation
        trade_data = {
            # Core position info
            "side": "LONG" if signal == "LONG_SPREAD" else "SHORT",
            "q1": qty_y,
            "q2": qty_x,
            # Phase 6.3: Full trade documentation
            "entry_time": datetime.now().isoformat(),
            "entry_zscore": round(entry_zscore, 2),
            "entry_price_y": round(price_y, 2),
            "entry_price_x": round(price_x, 2),
            "beta": round(hedge_ratio, 4),
            "intercept": round(intercept, 4),
            "target_zscore": 1.0,   # Exit threshold
            "stop_zscore": 3.0,     # Stop loss threshold
            "lot_y": lot_y,
            "lot_x": lot_x,
        }
        
        if signal == "LONG_SPREAD":
            print(f"   🚀 ENTRY LONG: Buy {qty_y} {s1}, Sell {qty_x} {s2} | Z={entry_zscore:.2f}")
            self.executor.place_pair_order(
                s1, "BUY", qty_y, price_y,
                s2, "SELL", qty_x, price_x,
                product=PRODUCT_TYPE
            )
            self.active_trades[pair_key] = trade_data
            self.state_manager.save(self.active_trades)  # Persist (Optimization #4)
            
        elif signal == "SHORT_SPREAD":
            print(f"   🚀 ENTRY SHORT: Sell {qty_y} {s1}, Buy {qty_x} {s2} | Z={entry_zscore:.2f}")
            self.executor.place_pair_order(
                s1, "SELL", qty_y, price_y,
                s2, "BUY", qty_x, price_x,
                product=PRODUCT_TYPE
            )
            self.active_trades[pair_key] = trade_data
            self.state_manager.save(self.active_trades)  # Persist (Optimization #4)
    
    def _close_position(self, pair_key: str, s1: str, s2: str, px1: float, px2: float, 
                        exit_reason: str = "MANUAL"):
        """
        Close position with full exit documentation (Phase 8).
        
        Calculates final P&L and logs exit details.
        """
        if pair_key not in self.active_trades:
            return
        
        trade = self.active_trades[pair_key]
        
        # Phase 8: Calculate Final P&L
        entry_y = trade.get('entry_price_y', px1)
        entry_x = trade.get('entry_price_x', px2)
        qty_y = trade.get('q1', 0)
        qty_x = trade.get('q2', 0)
        side = trade.get('side', 'LONG')
        
        if side == "LONG":
            # Long spread: Long Y, Short X
            pnl_y = (px1 - entry_y) * qty_y
            pnl_x = (entry_x - px2) * qty_x
        else:
            # Short spread: Short Y, Long X
            pnl_y = (entry_y - px1) * qty_y
            pnl_x = (px2 - entry_x) * qty_x
        
        realized_pnl = pnl_y + pnl_x
        pnl_symbol = "📈" if realized_pnl > 0 else "📉"
        
        print(f"   {pnl_symbol} CLOSING: {pair_key} | P&L: ₹{realized_pnl:,.0f} (Y:₹{pnl_y:,.0f}, X:₹{pnl_x:,.0f}) | {exit_reason}")
        
        # Execute close orders
        if trade['side'] == "LONG":
            self.executor.place_pair_order(s1, "SELL", trade['q1'], px1, s2, "BUY", trade['q2'], px2, product=PRODUCT_TYPE)
        else:
            self.executor.place_pair_order(s1, "BUY", trade['q1'], px1, s2, "SELL", trade['q2'], px2, product=PRODUCT_TYPE)
        
        # Phase 8: Document exit details to trades DB
        self.executor.log_trade(s1, "EXIT_" + ("SELL" if side == "LONG" else "BUY"), qty_y, px1, "StatArb")
        self.executor.log_trade(s2, "EXIT_" + ("BUY" if side == "LONG" else "SELL"), qty_x, px2, "StatArb")
        
        # Log complete trade summary
        entry_time = trade.get('entry_time', '')
        entry_zscore = trade.get('entry_zscore', 0)
        print(f"   📋 TRADE COMPLETE: Entry Z={entry_zscore:.2f} → Exit | Held from {entry_time[:10] if entry_time else 'N/A'}")
        
        del self.active_trades[pair_key]
        self.state_manager.save(self.active_trades)  # Persist (Optimization #4)
    
    def _is_market_open(self) -> bool:
        """
        Check if market is open for trading (Checklist Phase 4.3).
        
        NSE Trading Hours: 9:15 AM - 3:30 PM IST, Monday-Friday
        """
        now = datetime.now()
        
        # Check weekday (0=Monday, 6=Sunday)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check time (NSE: 9:15 AM to 3:30 PM)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
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
