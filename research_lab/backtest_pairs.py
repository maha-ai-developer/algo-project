"""
Professional Backtest v4.0 - Hybrid Data Model

The "Hybrid" approach for realistic Statistical Arbitrage:
- Dataset A (Spot): For signals (Cointegration, Z-Score, Moving Averages)
- Dataset B (Futures): For P&L (Entry/Exit prices, Margin, Slippage)

Why:
- Spot prices are "clean" - no expiry gaps or time decay noise
- Futures prices reflect actual execution reality
- Captures true basis risk and liquidity conditions
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import time
import threading
from datetime import datetime, date
from tabulate import tabulate
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from strategies.stat_arb_bot import StatArbBot
from strategies.guardian import AssumptionGuardian
from infrastructure.data.futures_utils import (
    get_lot_size, 
    calculate_margin_required,
    get_current_month_future,
)


# ============================================================
# CONFIGURATION (Document-Compliant + Futures-Ready)
# ============================================================

# Strategy Parameters (per paper-maharajan.md)
# Requires 400+ data points for meaningful backtest
# scan_pairs.py now downloads 730 days (2 years)
LOOKBACK_WINDOW = 250        # Rolling window for Z-score
Z_ENTRY_THRESHOLD = 2.0      # Entry when |Z| > 2.0
Z_EXIT_THRESHOLD = 0.5       # Exit when |Z| < 0.5
Z_STOP_THRESHOLD = 4.0       # Emergency stop when |Z| > 4.0
MAX_HOLDING_DAYS = 10        # Max holding period

# Guardian Control
ENABLE_GUARDIAN = False      # Set True to enable Guardian assumption monitoring

# Futures Trading Costs (Zerodha NRML)
BROKERAGE_PER_ORDER = 20     # Flat ₹20 per executed order
STT_PCT = 0.0001             # 0.01% on sell side (futures)
EXCHANGE_TXN_PCT = 0.00053   # NSE transaction charges
GST_PCT = 0.18               # 18% GST on brokerage + txn charges
SEBI_CHARGES = 0.000001      # SEBI charges per turnover
STAMP_DUTY_PCT = 0.00002     # Stamp duty
SLIPPAGE_PCT = 0.0005        # Market impact slippage (0.05%)

# Capital & Margin
DEFAULT_CAPITAL = 500000     # ₹5 lakh for futures trading
MAX_LOTS_PER_LEG = 5         # Maximum lots per leg (risk control)
MARGIN_BUFFER_PCT = 0.20     # 20% buffer on margin (for M2M)


# ============================================================
# PROGRESS MANAGER
# ============================================================

class BacktestProgressManager:
    """Saves intermediate progress for crash recovery."""
    
    def __init__(self, progress_file: Optional[str] = None):
        self.progress_file = progress_file or os.path.join(config.DATA_DIR, "backtest_progress.json")
        self._lock = threading.Lock()
        self.results: List[Dict] = []
        self.tested_pairs: set = set()
    
    def _pair_key(self, leg1: str, leg2: str) -> str:
        return f"{leg1}-{leg2}"
    
    def load(self) -> tuple:
        if not os.path.exists(self.progress_file):
            return [], set()
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            results = data.get('results', [])
            tested = set(data.get('tested_pairs', []))
            print(f"   📂 Resuming: {len(tested)} pairs already tested")
            return results, tested
        except (json.JSONDecodeError, ValueError):
            return [], set()
    
    def save(self):
        with self._lock:
            try:
                with open(self.progress_file, 'w') as f:
                    json.dump({
                        'results': self.results,
                        'tested_pairs': list(self.tested_pairs),
                        '_updated_at': datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                print(f"   ⚠️ Progress save failed: {e}")
    
    def add_result(self, leg1: str, leg2: str, result: Dict):
        key = self._pair_key(leg1, leg2)
        with self._lock:
            self.results.append(result)
            self.tested_pairs.add(key)
        if len(self.results) % 10 == 0:
            self.save()
    
    def is_tested(self, leg1: str, leg2: str) -> bool:
        return self._pair_key(leg1, leg2) in self.tested_pairs
    
    def clear(self):
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)


# ============================================================
# HYBRID BACKTEST ENGINE
# ============================================================

class HybridBacktest:
    """
    Hybrid backtest engine with dual dataset approach.
    
    - Dataset A (Spot): For signal generation (clean prices)
    - Dataset B (Futures): For P&L calculation (execution prices)
    
    Key features:
    - Lot size constraints (actual NSE lot sizes)
    - Margin-based capital allocation
    - Realistic futures transaction costs
    - Basis risk tracking
    """
    
    def __init__(self, capital: float = DEFAULT_CAPITAL):
        self.capital = capital
        self.available_margin = capital
    
    def run(self, pair_data: Dict) -> Dict:
        """Run hybrid backtest for a single pair."""
        y_sym = pair_data['leg1']
        x_sym = pair_data['leg2']
        initial_beta = pair_data['hedge_ratio']
        intercept = pair_data.get('intercept', 0.0)
        sector = pair_data.get('sector', 'UNKNOWN')
        
        # Get lot sizes
        lot_y = get_lot_size(y_sym)
        lot_x = get_lot_size(x_sym)
        
        # Initialize strategy components
        bot = StatArbBot(
            entry_z=Z_ENTRY_THRESHOLD,
            exit_z=Z_EXIT_THRESHOLD,
            stop_z=Z_STOP_THRESHOLD,
            lookback=LOOKBACK_WINDOW
        )
        bot.beta = initial_beta
        bot.intercept = intercept
        
        guardian = AssumptionGuardian(lookback_window=60)
        guardian.calibrate(initial_beta)
        
        # Load hybrid data
        df_spot, df_futures, data_info = self._load_hybrid_data(y_sym, x_sym)
        
        if df_spot is None:
            return {'error': 'Spot data not found', 'pair': f"{y_sym}-{x_sym}"}
        
        # Minimum: LOOKBACK_WINDOW days (reduced from +50 buffer)
        if len(df_spot) < LOOKBACK_WINDOW:
            return {'error': f'Insufficient data ({len(df_spot)} rows, need {LOOKBACK_WINDOW})', 'pair': f"{y_sym}-{x_sym}"}
        
        # State variables
        position = 0  # 0=flat, 1=long spread, -1=short spread
        entry_spot_y, entry_spot_x = 0.0, 0.0
        entry_fut_y, entry_fut_x = 0.0, 0.0
        lots_y, lots_x = 0, 0
        entry_date = None
        equity = self.capital
        
        # Tracking
        trade_log = []
        halt_days = 0
        recalibrations = 0
        holding_days = 0
        margin_used = 0.0
        basis_risk_total = 0.0
        
        # History buffers (SPOT only - for signals)
        hist_y = []
        hist_x = []
        
        # Main backtest loop
        for i in range(len(df_spot)):
            dt = df_spot.index[i]
            
            # SPOT prices (for signals)
            spot_y = df_spot['Y'].iloc[i]
            spot_x = df_spot['X'].iloc[i]
            
            # FUTURES prices (for P&L) - use spot as fallback
            if df_futures is not None and dt in df_futures.index:
                fut_y = df_futures['Y'].iloc[df_futures.index.get_loc(dt)]
                fut_x = df_futures['X'].iloc[df_futures.index.get_loc(dt)]
            else:
                fut_y = spot_y  # Fallback to spot
                fut_x = spot_x
            
            hist_y.append(spot_y)
            hist_x.append(spot_x)
            
            # Guardian health check (uses SPOT data)
            # Bypass if disabled for testing
            if ENABLE_GUARDIAN:
                guardian.update_data(spot_y, spot_x)
                status, reason = guardian.diagnose()
                
                # Auto-recalibration
                if guardian.needs_recalibration():
                    new_beta = guardian.force_recalibrate_to_current()
                    if new_beta:
                        bot.beta = new_beta
                        recalibrations += 1
                        status = "YELLOW"
            else:
                status, reason = "GREEN", "Guardian Disabled"
            
            # RED = forced exit & halt
            if status == "RED":
                halt_days += 1
                if position != 0:
                    # Exit at FUTURES prices
                    pnl = self._calc_futures_pnl(
                        position, entry_fut_y, entry_fut_x, fut_y, fut_x, 
                        lots_y, lots_x, lot_y, lot_x
                    )
                    equity += pnl
                    
                    # Track basis risk
                    basis_y = abs(fut_y - spot_y) / spot_y * 100
                    basis_x = abs(fut_x - spot_x) / spot_x * 100
                    basis_risk_total += (basis_y + basis_x) / 2
                    
                    trade_log.append({
                        "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                        "type": "GUARDIAN_HALT",
                        "pnl": round(pnl, 2),
                        "reason": reason,
                        "basis_y": round(basis_y, 2),
                        "basis_x": round(basis_x, 2)
                    })
                    position = 0
                    holding_days = 0
                    margin_used = 0
                continue
            
            # Need enough data for rolling Z-score
            if len(hist_y) < LOOKBACK_WINDOW:
                continue
            
            # Calculate rolling Z-score (SPOT data - NO LOOK-AHEAD BIAS)
            window_y = np.array(hist_y[-LOOKBACK_WINDOW:])
            window_x = np.array(hist_x[-LOOKBACK_WINDOW:])
            spread = window_y - (bot.beta * window_x + bot.intercept)
            z_mean = np.mean(spread)
            z_std = np.std(spread) if np.std(spread) > 0 else 1.0
            
            curr_spread = spot_y - (bot.beta * spot_x + bot.intercept)
            z = (curr_spread - z_mean) / z_std
            
            # Position management
            if position != 0:
                holding_days += 1
                
                # Mark-to-Market P&L (FUTURES prices)
                mtm_pnl = self._calc_futures_pnl(
                    position, entry_fut_y, entry_fut_x, fut_y, fut_x,
                    lots_y, lots_x, lot_y, lot_x
                )
                
                # Exit conditions (based on SPOT Z-score)
                take_profit = (position == 1 and z > -Z_EXIT_THRESHOLD) or \
                              (position == -1 and z < Z_EXIT_THRESHOLD)
                stop_loss = abs(z) > Z_STOP_THRESHOLD
                time_stop = holding_days >= MAX_HOLDING_DAYS
                margin_call = (equity + mtm_pnl) < margin_used * 0.5
                
                if take_profit or stop_loss or time_stop or margin_call:
                    # Exit at FUTURES prices
                    pnl = self._calc_futures_pnl(
                        position, entry_fut_y, entry_fut_x, fut_y, fut_x,
                        lots_y, lots_x, lot_y, lot_x
                    )
                    equity += pnl
                    
                    # Track basis risk
                    basis_y = abs(fut_y - spot_y) / spot_y * 100
                    basis_x = abs(fut_x - spot_x) / spot_x * 100
                    basis_risk_total += (basis_y + basis_x) / 2
                    
                    exit_type = "TP" if take_profit else ("SL" if stop_loss else ("TIME" if time_stop else "MARGIN"))
                    trade_log.append({
                        "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                        "type": f"EXIT_{exit_type}",
                        "pnl": round(pnl, 2),
                        "z": round(z, 2),
                        "days_held": holding_days,
                        "basis_y": round(basis_y, 2),
                        "basis_x": round(basis_x, 2)
                    })
                    position = 0
                    holding_days = 0
                    margin_used = 0
            
            # Entry conditions (only if flat and GREEN)
            if position == 0 and status == "GREEN":
                # Calculate position size based on FUTURES margin
                lots_y, lots_x, required_margin = self._calculate_position_size(
                    fut_y, fut_x, lot_y, lot_x, bot.beta, equity
                )
                
                if lots_y > 0 and lots_x > 0 and required_margin < equity * 0.8:
                    if z < -Z_ENTRY_THRESHOLD:
                        # Long spread: Buy Y futures, Sell X futures
                        position = 1
                        entry_spot_y, entry_spot_x = spot_y, spot_x
                        entry_fut_y, entry_fut_x = fut_y, fut_x  # Entry at FUTURES price
                        entry_date = dt
                        margin_used = required_margin
                        equity -= self._entry_costs(fut_y, fut_x, lots_y, lots_x, lot_y, lot_x)
                        
                        # Log entry with basis
                        basis_y = abs(fut_y - spot_y) / spot_y * 100
                        basis_x = abs(fut_x - spot_x) / spot_x * 100
                        
                        trade_log.append({
                            "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                            "type": "ENTRY_LONG",
                            "z": round(z, 2),
                            "lots_y": lots_y,
                            "lots_x": lots_x,
                            "margin": round(required_margin, 2),
                            "basis_y": round(basis_y, 2),
                            "basis_x": round(basis_x, 2)
                        })
                        
                    elif z > Z_ENTRY_THRESHOLD:
                        # Short spread: Sell Y futures, Buy X futures
                        position = -1
                        entry_spot_y, entry_spot_x = spot_y, spot_x
                        entry_fut_y, entry_fut_x = fut_y, fut_x
                        entry_date = dt
                        margin_used = required_margin
                        equity -= self._entry_costs(fut_y, fut_x, lots_y, lots_x, lot_y, lot_x)
                        
                        basis_y = abs(fut_y - spot_y) / spot_y * 100
                        basis_x = abs(fut_x - spot_x) / spot_x * 100
                        
                        trade_log.append({
                            "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                            "type": "ENTRY_SHORT",
                            "z": round(z, 2),
                            "lots_y": lots_y,
                            "lots_x": lots_x,
                            "margin": round(required_margin, 2),
                            "basis_y": round(basis_y, 2),
                            "basis_x": round(basis_x, 2)
                        })
        
        # Final stats
        total_return = ((equity - self.capital) / self.capital) * 100
        trades = [t for t in trade_log if t['type'].startswith('EXIT')]
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        avg_basis = basis_risk_total / max(len(trades), 1)
        
        return {
            "pair": f"{y_sym}-{x_sym}",
            "leg1": y_sym,
            "leg2": x_sym,
            "sector": sector,
            "lot_size_y": lot_y,
            "lot_size_x": lot_x,
            "data_mode": data_info,
            "return_pct": round(total_return, 2),
            "trades": len(trades),
            "win_rate": round(len(winning_trades) / max(len(trades), 1) * 100, 1),
            "avg_basis_risk": round(avg_basis, 2),
            "halt_days": halt_days,
            "recalibrations": recalibrations,
            "final_beta": round(bot.beta, 4),
            "final_intercept": round(bot.intercept, 4),
            "avg_holding_days": round(np.mean([t.get('days_held', 0) for t in trades]) if trades else 0, 1)
        }
    
    def _load_hybrid_data(self, y_sym: str, x_sym: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
        """
        Load hybrid datasets:
        - Dataset A (Spot): For signals
        - Dataset B (Futures): For P&L (optional)
        
        Returns:
            (spot_df, futures_df, data_info_string)
        """
        # SPOT DATA (Required)
        spot_path_y = os.path.join(config.DATA_DIR, f"{y_sym}_day.csv")
        spot_path_x = os.path.join(config.DATA_DIR, f"{x_sym}_day.csv")
        
        if not os.path.exists(spot_path_y) or not os.path.exists(spot_path_x):
            return None, None, "NO_DATA"
        
        try:
            df_spot_y = pd.read_csv(spot_path_y)
            df_spot_x = pd.read_csv(spot_path_x)
            df_spot_y['date'] = pd.to_datetime(df_spot_y['date'])
            df_spot_x['date'] = pd.to_datetime(df_spot_x['date'])
            df_spot_y.set_index('date', inplace=True)
            df_spot_x.set_index('date', inplace=True)
            df_spot = pd.concat([df_spot_y['close'], df_spot_x['close']], axis=1).dropna()
            df_spot.columns = ['Y', 'X']
        except Exception as e:
            return None, None, f"SPOT_ERROR: {e}"
        
        # FUTURES DATA (Optional - for P&L)
        futures_y = get_current_month_future(y_sym)
        futures_x = get_current_month_future(x_sym)
        
        futures_path_y = os.path.join(config.DATA_DIR, f"{futures_y}_day.csv")
        futures_path_x = os.path.join(config.DATA_DIR, f"{futures_x}_day.csv")
        
        df_futures = None
        
        if os.path.exists(futures_path_y) and os.path.exists(futures_path_x):
            try:
                df_fut_y = pd.read_csv(futures_path_y)
                df_fut_x = pd.read_csv(futures_path_x)
                df_fut_y['date'] = pd.to_datetime(df_fut_y['date'])
                df_fut_x['date'] = pd.to_datetime(df_fut_x['date'])
                df_fut_y.set_index('date', inplace=True)
                df_fut_x.set_index('date', inplace=True)
                df_futures = pd.concat([df_fut_y['close'], df_fut_x['close']], axis=1).dropna()
                df_futures.columns = ['Y', 'X']
                return df_spot, df_futures, "HYBRID (Spot+Futures)"
            except Exception:
                pass
        
        # Check for continuous futures data pattern
        # Look for any futures files matching the symbol
        futures_files_y = [f for f in os.listdir(config.DATA_DIR) if f.startswith(y_sym) and 'FUT' in f]
        futures_files_x = [f for f in os.listdir(config.DATA_DIR) if f.startswith(x_sym) and 'FUT' in f]
        
        if futures_files_y and futures_files_x:
            try:
                # Use the first available futures file
                df_fut_y = pd.read_csv(os.path.join(config.DATA_DIR, futures_files_y[0]))
                df_fut_x = pd.read_csv(os.path.join(config.DATA_DIR, futures_files_x[0]))
                df_fut_y['date'] = pd.to_datetime(df_fut_y['date'])
                df_fut_x['date'] = pd.to_datetime(df_fut_x['date'])
                df_fut_y.set_index('date', inplace=True)
                df_fut_x.set_index('date', inplace=True)
                df_futures = pd.concat([df_fut_y['close'], df_fut_x['close']], axis=1).dropna()
                df_futures.columns = ['Y', 'X']
                return df_spot, df_futures, "HYBRID (Spot+Futures)"
            except Exception:
                pass
        
        # Fallback: Spot only (P&L at spot prices - less accurate)
        return df_spot, None, "SPOT_ONLY (proxy P&L)"
    
    def _calculate_position_size(self, price_y: float, price_x: float, 
                                  lot_y: int, lot_x: int, beta: float,
                                  available_capital: float) -> Tuple[int, int, float]:
        """Calculate lot-based position size with margin constraints."""
        # Estimate margin per lot (15% of contract value)
        margin_y_per_lot = price_y * lot_y * 0.15
        margin_x_per_lot = price_x * lot_x * 0.15
        
        # Total margin for 1 lot each
        total_margin = margin_y_per_lot + margin_x_per_lot
        
        # Calculate affordable lots
        max_affordable_lots = int(available_capital * (1 - MARGIN_BUFFER_PCT) / max(total_margin, 1))
        
        # Apply constraints
        lots = min(1, max_affordable_lots, MAX_LOTS_PER_LEG)
        lots = max(lots, 1)
        
        margin_required = (margin_y_per_lot + margin_x_per_lot) * lots
        
        return lots, lots, margin_required
    
    def _calc_futures_pnl(self, position: int, entry_y: float, entry_x: float,
                          exit_y: float, exit_x: float, lots_y: int, lots_x: int,
                          lot_size_y: int, lot_size_x: int) -> float:
        """Calculate futures P&L with lot sizes."""
        qty_y = lots_y * lot_size_y
        qty_x = lots_x * lot_size_x
        
        if position == 1:  # Long spread: Long Y, Short X
            pnl_y = (exit_y - entry_y) * qty_y
            pnl_x = (entry_x - exit_x) * qty_x
        else:  # Short spread: Short Y, Long X
            pnl_y = (entry_y - exit_y) * qty_y
            pnl_x = (exit_x - entry_x) * qty_x
        
        gross_pnl = pnl_y + pnl_x
        costs = self._exit_costs(exit_y, exit_x, lots_y, lots_x, lot_size_y, lot_size_x)
        
        return gross_pnl - costs
    
    def _entry_costs(self, price_y: float, price_x: float, lots_y: int, lots_x: int,
                     lot_size_y: int, lot_size_x: int) -> float:
        """Calculate entry transaction costs for futures."""
        turnover_y = price_y * lots_y * lot_size_y
        turnover_x = price_x * lots_x * lot_size_x
        total_turnover = turnover_y + turnover_x
        
        brokerage = BROKERAGE_PER_ORDER * 2
        exchange_txn = total_turnover * EXCHANGE_TXN_PCT
        sebi = total_turnover * SEBI_CHARGES
        stamp = total_turnover * STAMP_DUTY_PCT
        gst = (brokerage + exchange_txn) * GST_PCT
        slippage = total_turnover * SLIPPAGE_PCT
        
        return brokerage + exchange_txn + sebi + stamp + gst + slippage
    
    def _exit_costs(self, price_y: float, price_x: float, lots_y: int, lots_x: int,
                    lot_size_y: int, lot_size_x: int) -> float:
        """Calculate exit transaction costs (includes STT on sell)."""
        turnover_y = price_y * lots_y * lot_size_y
        turnover_x = price_x * lots_x * lot_size_x
        total_turnover = turnover_y + turnover_x
        
        brokerage = BROKERAGE_PER_ORDER * 2
        stt = total_turnover * STT_PCT
        exchange_txn = total_turnover * EXCHANGE_TXN_PCT
        sebi = total_turnover * SEBI_CHARGES
        stamp = total_turnover * STAMP_DUTY_PCT
        gst = (brokerage + exchange_txn) * GST_PCT
        slippage = total_turnover * SLIPPAGE_PCT
        
        return brokerage + stt + exchange_txn + sebi + stamp + gst + slippage


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_pro_backtest(resume: bool = True):
    """
    Run hybrid backtest on all candidate pairs.
    
    Uses:
    - Spot data for signal generation (clean prices)
    - Futures data for P&L calculation (if available)
    """
    print("--- 🧪 HYBRID BACKTEST v4.0 (Spot Signals + Futures P&L) ---")
    print(f"   📊 Lookback: {LOOKBACK_WINDOW} days | Z: ±{Z_ENTRY_THRESHOLD}/±{Z_EXIT_THRESHOLD}")
    print(f"   💰 Capital: ₹{DEFAULT_CAPITAL:,} | Max hold: {MAX_HOLDING_DAYS} days")
    print(f"   📦 Mode: SPOT for signals, FUTURES for P&L")
    
    # Load pairs from candidates file
    if not os.path.exists(config.PAIRS_CANDIDATES_FILE):
        print(f"❌ No candidates found at {config.PAIRS_CANDIDATES_FILE}")
        print("   Run 'python cli.py scan_pairs' first.")
        return
    
    with open(config.PAIRS_CANDIDATES_FILE, "r") as f:
        candidates = json.load(f)
    
    print(f"\n💼 Testing {len(candidates)} candidate pairs from pairs_candidates.json...")
    
    # Show sample pairs
    print("\n📋 Sample pairs:")
    for p in candidates[:3]:
        print(f"   {p['leg1']} ↔ {p['leg2']} ({p.get('sector', 'N/A')}) | β={p['hedge_ratio']:.3f}")
    
    engine = HybridBacktest()
    progress = BacktestProgressManager()
    
    if resume:
        progress.results, progress.tested_pairs = progress.load()
        candidates = [c for c in candidates if not progress.is_tested(c['leg1'], c['leg2'])]
        print(f"   📊 Remaining: {len(candidates)} pairs to test")
    
    if not candidates and progress.results:
        print("   ✅ All pairs already tested.")
        results = progress.results
    else:
        results = list(progress.results)
        start_time = time.time()
        
        for i, pair in enumerate(candidates, 1):
            leg1, leg2 = pair['leg1'], pair['leg2']
            
            pct = (i / len(candidates)) * 100
            sys.stdout.write(f"\r   👉 [{i}/{len(candidates)}] ({pct:.0f}%) {leg1}-{leg2}...     ")
            sys.stdout.flush()
            
            res = engine.run(pair)
            
            if 'error' not in res:
                results.append(res)
                progress.add_result(leg1, leg2, res)
            else:
                progress.tested_pairs.add(progress._pair_key(leg1, leg2))
        
        elapsed = time.time() - start_time
        print(f"\n\n   ⏱️ Completed in {elapsed:.1f}s")
        progress.save()
    
    if not results:
        print("❌ No valid results.")
        return
    
    # Display data mode distribution
    df_all = pd.DataFrame(results)
    print("\n📊 Data Mode Distribution:")
    print(df_all['data_mode'].value_counts().to_string())
    
    # Filter winners
    winners = df_all[
        (df_all['return_pct'] > 2.0) & 
        (df_all['win_rate'] > 45) & 
        (df_all['halt_days'] < 150)
    ].copy()
    winners = winners.sort_values(by='return_pct', ascending=False)
    
    print("\n🏆 TOP PERFORMING PAIRS (HYBRID BACKTEST)")
    display_cols = ['pair', 'return_pct', 'win_rate', 'trades', 'avg_basis_risk', 'data_mode']
    print(tabulate(
        winners[display_cols].head(15),
        headers=['Pair', 'Return %', 'Win %', 'Trades', 'Basis%', 'Data Mode'],
        tablefmt="simple_grid"
    ))
    
    # Save live config
    if not winners.empty:
        live_config = []
        for _, row in winners.head(10).iterrows():
            live_config.append({
                "leg1": row['leg1'],
                "leg2": row['leg2'],
                "sector": row.get('sector', 'UNKNOWN'),
                "hedge_ratio": row['final_beta'],
                "intercept": row['final_intercept'],
                "lot_size_y": row['lot_size_y'],
                "lot_size_x": row['lot_size_x'],
                "backtest_return": row['return_pct'],
                "strategy": "StatArb_Hybrid_v4"
            })
        
        with open(config.PAIRS_CONFIG, "w") as f:
            json.dump(live_config, f, indent=4)
        
        print(f"\n✅ Saved {len(live_config)} pairs to {config.PAIRS_CONFIG}")
        print("🚀 Run: python cli.py engine --mode PAPER")
        
        progress.clear()
    else:
        print("\n❌ No pairs met criteria.")


def run_pro_backtest_fresh():
    """Fresh backtest without resume."""
    run_pro_backtest(resume=False)


if __name__ == "__main__":
    run_pro_backtest()
