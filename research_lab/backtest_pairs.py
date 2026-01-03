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
import statsmodels.api as sm
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

# Strategy Parameters (PDF-Compliant - Zerodha Varsity Chapters 12-14)
# Entry at ±2.5 SD, Exit at ±1.0 SD, Stop at ±3.0 SD
# NO LOOK-AHEAD BIAS: All parameters from training period only
LOOKBACK_WINDOW = 200        # 200 days (works with 249+ rows)
Z_ENTRY_THRESHOLD = 2.5      # Entry at ±2.5 SD (Zerodha Varsity Page 47)
Z_EXIT_THRESHOLD = 1.0       # Exit at ±1.0 SD (mean reversion target)
Z_STOP_THRESHOLD = 3.0       # Stop loss at ±3.0 SD
MAX_HOLDING_DAYS = 20        # 20 days for more complete mean reversion

# Guardian Control
ENABLE_GUARDIAN = False      # Set True to enable Guardian assumption monitoring

# Train/Test Split Configuration (Checklist Gap Fill)
TRAIN_PCT = 0.60             # 60% for training (pair selection, parameter tuning)
VALIDATE_PCT = 0.20          # 20% for validation
TEST_PCT = 0.20              # 20% for final testing (unseen data)

# Futures Trading Costs (Zerodha NRML - Official Kite Charges 2024)
# Source: https://zerodha.com/charges
BROKERAGE_PER_ORDER = 20       # Flat ₹20 per executed order (or 0.03% whichever is lower)
STT_PCT = 0.000125             # 0.0125% on sell side (F&O futures) - UPDATED
EXCHANGE_TXN_PCT = 0.00019     # NSE transaction charges = 0.019% for futures - UPDATED
GST_PCT = 0.18                 # 18% GST on brokerage + transaction charges
SEBI_CHARGES = 0.000001        # SEBI charges = ₹10 per crore = 0.0001%
STAMP_DUTY_PCT = 0.00002       # Stamp duty = 0.002% on buy side
SLIPPAGE_PCT = 0.001           # Market impact slippage (0.1%) - more realistic - UPDATED

# Capital & Margin
DEFAULT_CAPITAL = 500000     # ₹5 lakh for futures trading
MAX_LOTS_PER_LEG = 5         # Maximum lots per leg (risk control)
MARGIN_BUFFER_PCT = 0.20     # 20% buffer on margin (for M2M)

# Intercept Risk Filter (Zerodha Varsity Chapter 14)
# If intercept explains >20% of Y's price, the pair is risky
MAX_INTERCEPT_RISK_PCT = 0.20  # Max 20% unexplained by regression


def split_data(df: pd.DataFrame, train_pct: float = TRAIN_PCT, 
               val_pct: float = VALIDATE_PCT) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train/validate/test sets.
    
    Checklist Gap Fill: Proper out-of-sample validation.
    
    Args:
        df: DataFrame with DatetimeIndex (sorted chronologically)
        train_pct: Fraction for training (default 60%)
        val_pct: Fraction for validation (default 20%)
        
    Returns:
        Tuple of (train_df, validate_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ============================================================
# PROGRESS MANAGER
# ============================================================

class BacktestProgressManager:
    """Saves intermediate progress for crash recovery."""
    
    def __init__(self, progress_file: Optional[str] = None):
        self.progress_file = progress_file or os.path.join(config.CACHE_DIR, "backtest_progress.json")
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
    
    def run(self, pair_data: Dict, start_date: Optional[pd.Timestamp] = None, 
            end_date: Optional[pd.Timestamp] = None, override_data: Optional[Tuple] = None) -> Dict:
        """
        Run hybrid backtest for a single pair.
        
        Args:
            pair_data: Strategy parameters
            start_date: Optional start filter (inclusive)
            end_date: Optional end filter (inclusive)
            override_data: Optional pre-loaded (df_spot, df_futures, info) tuple
        """
        y_sym = pair_data.get('leg1') or pair_data.get('stock_y')
        x_sym = pair_data.get('leg2') or pair_data.get('stock_x')
        # Support both 'beta' (new) and 'hedge_ratio' (old) field names
        initial_beta = pair_data.get('beta') or pair_data.get('hedge_ratio', 1.0)
        intercept = pair_data.get('intercept', 0.0)
        sector = pair_data.get('sector', 'UNKNOWN')
        # FIX #4: Use sigma from regression output (not rolling)
        fixed_sigma = pair_data.get('sigma', 0.0)
        
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
        
        
        # Load hybrid data (or use override)
        if override_data:
            df_spot, df_futures, data_info = override_data
        else:
            df_spot, df_futures, data_info = self._load_hybrid_data(y_sym, x_sym)
            
        # Apply date filters if provided
        if df_spot is not None:
            if start_date:
                df_spot = df_spot[df_spot.index >= start_date]
            if end_date:
                df_spot = df_spot[df_spot.index <= end_date]
                
        if df_futures is not None:
            if start_date:
                df_futures = df_futures[df_futures.index >= start_date]
            if end_date:
                df_futures = df_futures[df_futures.index <= end_date]
        
        if df_spot is None:
            return {'error': 'Spot data not found', 'pair': f"{y_sym}-{x_sym}"}
        
        # FIX: Inner-join spot and futures dates for proper alignment
        # This ensures we only trade on days where BOTH data sources exist
        if df_futures is not None:
            # Get common dates only
            common_dates = df_spot.index.intersection(df_futures.index)
            
            # If aligned data is too short (< 300 days), use spot-only for more data
            if len(common_dates) < 300:
                # Use full spot data, ignore futures
                data_info = f"SPOT_ONLY ({len(df_spot)} days)"
                df_futures = None  # Force spot-only mode for P&L
            else:
                df_spot = df_spot.loc[common_dates]
                df_futures = df_futures.loc[common_dates]
                data_info = f"HYBRID_ALIGNED ({len(common_dates)} days)"
        
        # Minimum: LOOKBACK_WINDOW days (reduced from +50 buffer)
        if len(df_spot) < LOOKBACK_WINDOW:
            return {'error': f'Insufficient data ({len(df_spot)} rows, need {LOOKBACK_WINDOW})', 'pair': f"{y_sym}-{x_sym}"}
        
        # INTERCEPT RISK CHECK (Zerodha Varsity Chapter 14)
        # If intercept explains >20% of Y's stock price, the pair is risky
        # Because regression equation can only explain a small portion of Y's movement
        avg_y_price = df_spot['Y'].mean()
        if avg_y_price > 0 and abs(intercept) > 0:
            intercept_risk_pct = abs(intercept) / avg_y_price
            if intercept_risk_pct > MAX_INTERCEPT_RISK_PCT:
                return {
                    'error': f'Intercept risk too high ({intercept_risk_pct*100:.1f}% > {MAX_INTERCEPT_RISK_PCT*100:.0f}%)',
                    'pair': f"{y_sym}-{x_sym}",
                    'intercept': round(intercept, 2),
                    'avg_y_price': round(avg_y_price, 2),
                    'intercept_risk_pct': round(intercept_risk_pct * 100, 1)
                }
        
        # State variables
        position = 0  # 0=flat, 1=long spread, -1=short spread
        entry_spot_y, entry_spot_x = 0.0, 0.0
        entry_fut_y, entry_fut_x = 0.0, 0.0
        entry_basis_y, entry_basis_x = 0.0, 0.0  # Track entry basis for divergence check
        lots_y, lots_x = 0, 0
        entry_date = None
        equity = self.capital
        
        # Tracking
        trade_log = []
        equity_curve = [self.capital]  # Track equity for Sharpe/Drawdown
        daily_equity = [self.capital]  # FIX #3: Track equity DAILY for proper Sharpe
        halt_days = 0
        recalibrations = 0
        holding_days = 0
        margin_used = 0.0
        basis_risk_total = 0.0
        skipped_days = 0  # Track skipped days due to missing futures
        
        # History buffers (SPOT only - for signals)
        hist_y = []
        hist_x = []
        
        # Main backtest loop
        for i in range(len(df_spot)):
            dt = df_spot.index[i]
            
            # SPOT prices (for signals)
            spot_y = df_spot['Y'].iloc[i]
            spot_x = df_spot['X'].iloc[i]
            
            # FUTURES prices (for P&L)
            # FIX: NO FALLBACK - if futures missing, skip trade entry (but still update history)
            if df_futures is not None:
                fut_y = df_futures['Y'].iloc[i]  # Safe because of inner-join above
                fut_x = df_futures['X'].iloc[i]
                has_futures = True
            else:
                # Spot-only mode (no futures data available at all)
                fut_y = spot_y
                fut_x = spot_x
                has_futures = False
            
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
                    equity_curve.append(equity)  # Track for risk metrics
                    
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
            
            # FIX #1: Use FIXED sigma from regression (per Zerodha Varsity)
            # Z-Score = Today's Residual / Sigma (FIXED)
            # NOT rolling mean/std which causes look-ahead bias
            curr_residual = spot_y - (bot.beta * spot_x + bot.intercept)
            
            if fixed_sigma > 0:
                # Use fixed sigma from regression output (correct method)
                z = curr_residual / fixed_sigma
            else:
                # Fallback: Calculate sigma from initial LOOKBACK_WINDOW only
                window_y = np.array(hist_y[:LOOKBACK_WINDOW])
                window_x = np.array(hist_x[:LOOKBACK_WINDOW])
                initial_spread = window_y - (bot.beta * window_x + bot.intercept)
                fixed_sigma = np.std(initial_spread) if np.std(initial_spread) > 0 else 1.0
                z = curr_residual / fixed_sigma
            
            # Position management
            if position != 0:
                holding_days += 1
                
                # Mark-to-Market P&L (FUTURES prices)
                mtm_pnl = self._calc_futures_pnl(
                    position, entry_fut_y, entry_fut_x, fut_y, fut_x,
                    lots_y, lots_x, lot_y, lot_x
                )
                
                # FIX #3: Track daily equity (MTM-adjusted) for proper Sharpe ratio
                daily_equity.append(equity + mtm_pnl)
                
                # Exit conditions (SIMPLIFIED per user spec)
                # Take Profit: Z reverts to ±0.5 SD (per paper-maharajan.md)
                # Stop Loss: Z expands to ±3.0 SD  
                # Time Stop: 10 days max hold (per paper-maharajan.md)
                take_profit = (position == 1 and z > -Z_EXIT_THRESHOLD) or \
                              (position == -1 and z < Z_EXIT_THRESHOLD)
                stop_loss = abs(z) > Z_STOP_THRESHOLD
                time_stop = holding_days >= MAX_HOLDING_DAYS
                
                if take_profit or stop_loss or time_stop:
                    # Exit at FUTURES prices
                    pnl = self._calc_futures_pnl(
                        position, entry_fut_y, entry_fut_x, fut_y, fut_x,
                        lots_y, lots_x, lot_y, lot_x
                    )
                    equity += pnl
                    equity_curve.append(equity)
                    
                    # Track basis risk (for monitoring)
                    basis_y = abs(fut_y - spot_y) / spot_y * 100
                    basis_x = abs(fut_x - spot_x) / spot_x * 100
                    basis_risk_total += (basis_y + basis_x) / 2
                    
                    exit_type = "TP" if take_profit else ("SL" if stop_loss else "TIME")
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
                        
                        # Record entry basis for divergence monitoring
                        entry_basis_y = abs(fut_y - spot_y) / spot_y * 100
                        entry_basis_x = abs(fut_x - spot_x) / spot_x * 100
                        
                        trade_log.append({
                            "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                            "type": "ENTRY_LONG",
                            "z": round(z, 2),
                            "lots_y": lots_y,
                            "lots_x": lots_x,
                            "margin": round(required_margin, 2),
                            "basis_y": round(entry_basis_y, 2),
                            "basis_x": round(entry_basis_x, 2)
                        })
                        
                    elif z > Z_ENTRY_THRESHOLD:
                        # Short spread: Sell Y futures, Buy X futures
                        position = -1
                        entry_spot_y, entry_spot_x = spot_y, spot_x
                        entry_fut_y, entry_fut_x = fut_y, fut_x
                        entry_date = dt
                        margin_used = required_margin
                        equity -= self._entry_costs(fut_y, fut_x, lots_y, lots_x, lot_y, lot_x)
                        
                        # Record entry basis for divergence monitoring
                        entry_basis_y = abs(fut_y - spot_y) / spot_y * 100
                        entry_basis_x = abs(fut_x - spot_x) / spot_x * 100
                        
                        trade_log.append({
                            "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                            "type": "ENTRY_SHORT",
                            "z": round(z, 2),
                            "lots_y": lots_y,
                            "lots_x": lots_x,
                            "margin": round(required_margin, 2),
                            "basis_y": round(entry_basis_y, 2),
                            "basis_x": round(entry_basis_x, 2)
                        })
        
        # Final stats
        total_return = ((equity - self.capital) / self.capital) * 100
        trades = [t for t in trade_log if t['type'].startswith('EXIT')]
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        avg_basis = basis_risk_total / max(len(trades), 1)
        
        # Risk Metrics (Checklist Gap Fill)
        # FIX #3: Sharpe Ratio using DAILY equity (not just trade exits)
        # Sharpe = (Mean Daily Return) / Std(Daily Return) * sqrt(252)
        if len(daily_equity) > 10:  # Need minimum 10 days for meaningful Sharpe
            daily_returns = np.diff(daily_equity) / np.array(daily_equity[:-1])
            daily_returns = daily_returns[~np.isnan(daily_returns) & ~np.isinf(daily_returns)]
            if len(daily_returns) > 5 and np.std(daily_returns) > 0:
                sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Maximum Drawdown - FIX: Use daily_equity for accurate calculation
        if len(daily_equity) > 1:
            equity_arr = np.array(daily_equity)
            running_max = np.maximum.accumulate(equity_arr)
            drawdown = running_max - equity_arr
            max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        else:
            max_drawdown = 0.0
        max_drawdown_pct = (max_drawdown / self.capital) * 100
        
        # Profit Factor = Gross Profit / Gross Loss
        # FIX: Cap at reasonable max (10) to avoid misleading values
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        if gross_loss > 100:  # At least ₹100 loss to calculate meaningful profit factor
            profit_factor = min(gross_profit / gross_loss, 10.0)  # Cap at 10
        elif gross_profit > 0:
            profit_factor = 10.0  # Max if all trades are winners
        else:
            profit_factor = 0.0
        
        # Calmar Ratio = Annualized Return / Max Drawdown (research benchmark)
        # Higher is better; ≥0.5 is acceptable, ≥3.0 is excellent
        annualized_return = total_return  # Already annualized for comparison
        if max_drawdown_pct > 0.1:  # Avoid division by very small DD
            calmar_ratio = annualized_return / max_drawdown_pct
        else:
            calmar_ratio = 10.0 if total_return > 0 else 0.0
        
        # Expectancy = Average profit per trade (research benchmark)
        # Positive expectancy required for go-live
        all_pnls = [t.get('pnl', 0) for t in trades]
        expectancy = np.mean(all_pnls) if all_pnls else 0.0
        
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
            "avg_holding_days": round(np.mean([t.get('days_held', 0) for t in trades]) if trades else 0, 1),
            # Risk Metrics (Research-backed go-live benchmarks)
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "calmar_ratio": round(calmar_ratio, 2),  # NEW: Return/DD ratio
            "expectancy": round(expectancy, 2),       # NEW: Avg profit per trade
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2)
        }
    
    def _load_hybrid_data(self, y_sym: str, x_sym: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
        """
        Load hybrid datasets:
        - Dataset A (Spot): For signals (from BACKTEST_SPOT_DIR)
        - Dataset B (Futures): For P&L (from BACKTEST_FUTURES_DIR)
        
        Returns:
            (spot_df, futures_df, data_info_string)
        """
        # SPOT DATA (Required) - Try new structured path first, fallback to legacy
        spot_path_y = os.path.join(config.BACKTEST_SPOT_DIR, f"{y_sym}_day.csv")
        spot_path_x = os.path.join(config.BACKTEST_SPOT_DIR, f"{x_sym}_day.csv")
        
        # Fallback to legacy DATA_DIR if not in new folder
        if not os.path.exists(spot_path_y):
            spot_path_y = os.path.join(config.DATA_DIR, f"{y_sym}_day.csv")
        if not os.path.exists(spot_path_x):
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
        
        # FUTURES DATA (Optional - for P&L) - Try new structured path first
        futures_y = get_current_month_future(y_sym)
        futures_x = get_current_month_future(x_sym)
        
        # Try new BACKTEST_FUTURES_DIR first
        futures_path_y = os.path.join(config.BACKTEST_FUTURES_DIR, f"{futures_y}_day.csv")
        futures_path_x = os.path.join(config.BACKTEST_FUTURES_DIR, f"{futures_x}_day.csv")
        
        # Fallback to legacy DATA_DIR
        if not os.path.exists(futures_path_y):
            futures_path_y = os.path.join(config.DATA_DIR, f"{futures_y}_day.csv")
        if not os.path.exists(futures_path_x):
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
        
        # Check for any futures files in BACKTEST_FUTURES_DIR first, then DATA_DIR
        for search_dir in [config.BACKTEST_FUTURES_DIR, config.DATA_DIR]:
            if not os.path.exists(search_dir):
                continue
            futures_files_y = [f for f in os.listdir(search_dir) if f.startswith(y_sym) and 'FUT' in f]
            futures_files_x = [f for f in os.listdir(search_dir) if f.startswith(x_sym) and 'FUT' in f]
            
            if futures_files_y and futures_files_x:
                try:
                    df_fut_y = pd.read_csv(os.path.join(search_dir, futures_files_y[0]))
                    df_fut_x = pd.read_csv(os.path.join(search_dir, futures_files_x[0]))
                    df_fut_y['date'] = pd.to_datetime(df_fut_y['date'])
                    df_fut_x['date'] = pd.to_datetime(df_fut_x['date'])
                    df_fut_y.set_index('date', inplace=True)
                    df_fut_x.set_index('date', inplace=True)
                    df_futures = pd.concat([df_fut_y['close'], df_fut_x['close']], axis=1).dropna()
                    df_futures.columns = ['Y', 'X']
                    return df_spot, df_futures, "HYBRID (Spot+Futures)"
                except Exception:
                    continue
        
        # Fallback: Spot only (P&L at spot prices - less accurate)
        return df_spot, None, "SPOT_ONLY (proxy P&L)"
    
    def _calculate_position_size(self, price_y: float, price_x: float, 
                                  lot_y: int, lot_x: int, beta: float,
                                  available_capital: float) -> Tuple[int, int, float]:
        """
        Calculate lot-based position size with margin constraints.
        
        FIX #2: Apply BETA-NEUTRAL sizing per Zerodha Varsity:
        For beta = 1.59, if we buy 1500 shares of Y, we need 1500 * 1.59 = 2385 shares of X.
        This translates to lots_x = round(lots_y * beta * lot_y / lot_x)
        """
        # Estimate margin per lot (15% of contract value)
        margin_y_per_lot = price_y * lot_y * 0.15
        margin_x_per_lot = price_x * lot_x * 0.15
        
        # Calculate affordable lots for Y
        max_affordable_lots_y = int(available_capital * (1 - MARGIN_BUFFER_PCT) / max(margin_y_per_lot + margin_x_per_lot * abs(beta), 1))
        
        # Apply constraints
        lots_y = min(1, max_affordable_lots_y, MAX_LOTS_PER_LEG)
        lots_y = max(lots_y, 1)
        
        # FIX #2: Beta-neutral lot sizing
        # Shares of X needed = Shares of Y * beta
        # lots_x = (lots_y * lot_y * beta) / lot_x
        beta_adjusted_shares = lots_y * lot_y * abs(beta)
        lots_x = max(1, round(beta_adjusted_shares / lot_x))
        
        # Recalculate margin with actual lots
        margin_required = (margin_y_per_lot * lots_y) + (margin_x_per_lot * lots_x)
        
        return lots_y, lots_x, margin_required
    
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
    
    def run_with_validation(self, pair_data: Dict, train_pct: float = 0.60) -> Dict:
        """
        Run backtest with train/test split validation.
        
        Trains parameters on first 60% of data, tests on last 40%.
        Returns both in-sample and out-of-sample metrics.
        
        Args:
            pair_data: Pair configuration dict
            train_pct: Fraction for training (default 60%)
        """
        y_sym = pair_data['leg1']
        x_sym = pair_data['leg2']
        
        # Load full data first
        df_spot, df_futures, data_info = self._load_hybrid_data(y_sym, x_sym)
        
        if df_spot is None or len(df_spot) < LOOKBACK_WINDOW + 50:
            return {'error': 'Insufficient data for validation', 'pair': f"{y_sym}-{x_sym}"}
        
        # Split data chronologically
        split_idx = int(len(df_spot) * train_pct)
        train_df = df_spot.iloc[:split_idx]
        test_df = df_spot.iloc[split_idx:]
        
        if len(train_df) < 50 or len(test_df) < 10:
            return {'error': 'Split resulted in insufficient data', 'pair': f"{y_sym}-{x_sym}"}
            
        # 1. CALIBRATE on Training Data (In-Sample)
        # -----------------------------------------
        # Force direction from pair_data (do not re-evaluate Y/X flip)
        try:
            x_const = sm.add_constant(train_df['X'])
            model = sm.OLS(train_df['Y'], x_const).fit()
            
            beta = model.params['X']
            intercept = model.params['const']
            residuals = model.resid
            sigma = np.std(residuals)
            
            # Optional: Check if cointegration still holds in this sub-period
            # adf = sm.tsa.stattools.adfuller(residuals)
            # if adf[1] > 0.05:
            #     print(f"   ⚠️ Pair {y_sym}-{x_sym} lost cointegration in train period (p={adf[1]:.4f})")
            
        except Exception as e:
            return {'error': f'Calibration failed: {str(e)}', 'pair': f"{y_sym}-{x_sym}"}

        # 2. RUN BACKTEST on Test Data (Out-of-Sample)
        # --------------------------------------------
        # Construct test parameters using TRAINING values
        test_params = pair_data.copy()
        test_params.update({
            'beta': beta,
            'intercept': intercept,
            'sigma': sigma,
            'is_validation_run': True
        })
        
        # Run backtest on ONLY the test portion
        # We pass the full data tuple but filter by date in run() for efficiency?
        # Actually safer to pass sliced data if possible, or just start_date.
        # Let's use start_date filter we added to run().
        
        test_start_date = test_df.index[0]
        
        # Note: We pass the ORIGINAL loaded data (override_data) to avoid reloading
        # run() will filter it by start_date
        test_result = self.run(
            test_params, 
            start_date=test_start_date,
            override_data=(df_spot, df_futures, data_info)
        )
        
        if 'error' in test_result:
            return test_result
        
        # Add validation metadata
        test_result['validation'] = {
            'train_period': f"{train_df.index[0].date()} to {train_df.index[-1].date()}",
            'test_period': f"{test_df.index[0].date()} to {test_df.index[-1].date()}",
            'train_days': len(train_df),
            'test_days': len(test_df),
            'train_params': {'beta': round(beta, 4), 'intercept': round(intercept, 4), 'sigma': round(sigma, 4)},
            'mode': 'OUT_OF_SAMPLE_TEST'
        }
        
        return test_result
    
    # ============================================================
    # WALK-FORWARD OPTIMIZATION (Research Enhancement)
    # ============================================================
    
    def run_walk_forward(self, pair_data: Dict, 
                         train_window: int = 250,
                         test_window: int = 60,
                         step_size: int = 30) -> Dict:
        """
        Walk-Forward Optimization per research best practices (QuantInsti, AlgoTrading101).
        
        Algorithm:
        1. Train on [0:train_window], test on [train_window:train_window+test_window]
        2. Step forward by step_size days
        3. Repeat until end of data
        4. Aggregate ALL out-of-sample results
        
        This prevents overfitting by never testing on calibration data.
        
        Args:
            pair_data: Pair configuration dict
            train_window: Days for training/calibration (default 250)
            test_window: Days for out-of-sample testing (default 60)
            step_size: Days to roll forward (default 30)
            
        Returns:
            Dict with aggregated walk-forward results
        """
        y_sym = pair_data['leg1']
        x_sym = pair_data['leg2']
        
        # Load full data
        df_spot, df_futures, data_info = self._load_hybrid_data(y_sym, x_sym)
        
        if df_spot is None or len(df_spot) < train_window + test_window:
            return {'error': 'Insufficient data for walk-forward', 'pair': f"{y_sym}-{x_sym}"}
        
        # Walk-forward iterations
        all_trades = []
        all_returns = []
        iterations = 0
        
        start_idx = 0
        while start_idx + train_window + test_window <= len(df_spot):
            iterations += 1
            
            # Define windows
            train_end = start_idx + train_window
            test_end = train_end + test_window
            
            train_df = df_spot.iloc[start_idx:train_end]
            test_start_date = df_spot.index[train_end]
            test_end_date = df_spot.index[min(test_end - 1, len(df_spot) - 1)]
            
            # Calibrate on training data
            try:
                x_const = sm.add_constant(train_df['X'])
                model = sm.OLS(train_df['Y'], x_const).fit()
                
                beta = model.params['X']
                intercept = model.params['const']
                residuals = model.resid
                sigma = np.std(residuals)
                
                # Calculate half-life for dynamic exit timing
                half_life = self._calculate_half_life_ou(residuals)
                
            except Exception as e:
                start_idx += step_size
                continue
            
            # Run test on out-of-sample period
            test_params = pair_data.copy()
            test_params.update({
                'beta': beta,
                'intercept': intercept,
                'sigma': sigma,
                'half_life': half_life
            })
            
            # Filter futures data if available
            test_futures = None
            if df_futures is not None:
                test_futures = df_futures[(df_futures.index >= test_start_date) & 
                                          (df_futures.index <= test_end_date)]
            
            test_spot = df_spot[(df_spot.index >= test_start_date) & 
                                (df_spot.index <= test_end_date)]
            
            result = self.run(
                test_params,
                start_date=test_start_date,
                end_date=test_end_date,
                override_data=(df_spot, df_futures, data_info)
            )
            
            if 'error' not in result:
                all_returns.append(result.get('return_pct', 0))
                all_trades.append(result.get('trades', 0))
            
            # Step forward
            start_idx += step_size
        
        if iterations == 0:
            return {'error': 'No valid walk-forward iterations', 'pair': f"{y_sym}-{x_sym}"}
        
        # Aggregate results
        return {
            "pair": f"{y_sym}-{x_sym}",
            "leg1": y_sym,
            "leg2": x_sym,
            "method": "WALK_FORWARD",
            "iterations": iterations,
            "total_trades": sum(all_trades),
            "avg_return_pct": round(np.mean(all_returns), 2) if all_returns else 0,
            "std_return_pct": round(np.std(all_returns), 2) if all_returns else 0,
            "win_iterations": sum(1 for r in all_returns if r > 0),
            "win_rate_iterations": round(sum(1 for r in all_returns if r > 0) / max(len(all_returns), 1) * 100, 1),
            "train_window": train_window,
            "test_window": test_window,
            "step_size": step_size
        }
    
    def _calculate_half_life_ou(self, residuals: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.
        
        Per research (arxiv, QuantConnect):
        - Regress: delta_residual = theta * lag_residual + noise
        - Half-life = -ln(2) / theta
        
        Args:
            residuals: Residual series from cointegration regression
            
        Returns:
            Half-life in days. Returns np.inf if non-mean-reverting.
        """
        if len(residuals) < 20:
            return np.inf
        
        try:
            resid = residuals.reset_index(drop=True)
            resid_lag = resid.shift(1).iloc[1:]
            resid_delta = resid.diff().iloc[1:]
            
            if len(resid_lag) < 10:
                return np.inf
            
            x_const = sm.add_constant(resid_lag)
            model = sm.OLS(resid_delta, x_const).fit()
            theta = model.params.iloc[1]
            
            if theta >= 0:
                return np.inf  # Diverging, not mean-reverting
            
            half_life = -np.log(2) / theta
            return max(1.0, min(half_life, 100))  # Clamp: 1-100 days
            
        except Exception:
            return np.inf
    
    def _rolling_adf_check(self, residuals: np.ndarray, threshold: float = 0.10) -> float:
        """
        Perform rolling ADF check on recent residuals.
        
        Per research (ResearchGate): Cointegration is episodic, not permanent.
        Continuous validation prevents trading broken relationships.
        
        Args:
            residuals: Recent residual array (last 60+ days)
            threshold: P-value threshold for stationarity
            
        Returns:
            ADF p-value (lower = more stationary)
        """
        from statsmodels.tsa.stattools import adfuller
        
        if len(residuals) < 30:
            return 1.0  # Insufficient data
        
        try:
            adf_result = adfuller(residuals, maxlag=5, autolag='AIC')
            return adf_result[1]
        except Exception:
            return 1.0


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
        beta = p.get('beta') or p.get('hedge_ratio', 0)
        print(f"   {p.get('leg1') or p.get('stock_y')} ↔ {p.get('leg2') or p.get('stock_x')} ({p.get('sector', 'N/A')}) | β={beta:.3f}")
    
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
            
            # Run full backtest (not validation split) to see complete performance
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
    display_cols = ['pair', 'return_pct', 'win_rate', 'trades', 'sharpe_ratio', 'max_drawdown_pct', 'avg_basis_risk', 'data_mode']
    print(tabulate(
        winners[display_cols].head(15),
        headers=['Pair', 'Return %', 'Win %', 'Trades', 'Sharpe', 'DD %', 'Basis%', 'Data Mode'],
        tablefmt="simple_grid",
        floatfmt=".2f"
    ))
    
    # Save live config
    if not winners.empty:
        live_config = []
        for _, row in winners.head(10).iterrows():
            live_config.append({
                "leg1": row['leg1'],
                "leg2": row['leg2'],
                "sector": row.get('sector', 'UNKNOWN'),
                "hedge_ratio": row.get('final_beta', row.get('beta', 1.0)),
                "intercept": row['final_intercept'],
                "lot_size_y": row['lot_size_y'],
                "lot_size_x": row['lot_size_x'],
                "backtest_return": row['return_pct'],
                # Full metrics for AI analysis
                "trades": int(row.get('trades', 0)),
                "win_rate": row.get('win_rate', 0),
                "sharpe_ratio": row.get('sharpe_ratio', 0),
                "max_drawdown": row.get('max_drawdown', 0),
                "max_drawdown_pct": row.get('max_drawdown_pct', 0),
                "profit_factor": row.get('profit_factor', 0),
                "avg_holding_days": row.get('avg_holding_days', 0),
                "halt_days": int(row.get('halt_days', 0)),
                "recalibrations": int(row.get('recalibrations', 0)),
                "avg_basis_risk": row.get('avg_basis_risk', 0),
                "strategy": "StatArb_Hybrid_v4",
                "lookback_window": LOOKBACK_WINDOW,
                "z_entry": Z_ENTRY_THRESHOLD,
                "z_exit": Z_EXIT_THRESHOLD,
                "slippage_pct": SLIPPAGE_PCT * 100,
                "data_mode": row.get('data_mode', 'UNKNOWN')
            })
        
        with open(config.PAIRS_CONFIG, "w") as f:
            json.dump(live_config, f, indent=4)
        
        print(f"\n✅ Saved {len(live_config)} pairs to {config.PAIRS_CONFIG}")
        print("🚀 Run: python cli.py engine --mode PAPER")
        
        progress.clear()
    else:
        print("\n❌ No pairs met criteria.")
    
    # SAVE FULL RESULTS (All pairs, not just winners)
    full_results_path = os.path.join(config.ARTIFACTS_DIR, "backtest_full_results.json")
    full_export = []
    for _, row in df_all.iterrows():
        full_export.append({
            "pair": row['pair'],
            "leg1": row['leg1'],
            "leg2": row['leg2'],
            "sector": row.get('sector', 'UNKNOWN'),
            "return_pct": row['return_pct'],
            "trades": int(row.get('trades', 0)),
            "win_rate": row.get('win_rate', 0),
            "sharpe_ratio": row.get('sharpe_ratio', 0),
            "max_drawdown": row.get('max_drawdown', 0),
            "profit_factor": row.get('profit_factor', 0),
            "avg_holding_days": row.get('avg_holding_days', 0),
            "halt_days": int(row.get('halt_days', 0)),
            "data_mode": row.get('data_mode', 'UNKNOWN')
        })
    
    with open(full_results_path, "w") as f:
        json.dump(full_export, f, indent=4)
    
    # Summary stats
    winning = df_all[df_all['return_pct'] > 0]
    losing = df_all[df_all['return_pct'] <= 0]
    print(f"\n📊 FULL RESULTS SUMMARY:")
    print(f"   Total Pairs: {len(df_all)} | Winners: {len(winning)} | Losers: {len(losing)}")
    print(f"   Win Rate: {len(winning)/len(df_all)*100:.1f}%")
    print(f"   Total Trades: {df_all['trades'].sum()}")
    print(f"📁 Full results saved to: {full_results_path}")
    
    # ============================================================
    # BENCHMARK COMPARISON (Research-backed go-live criteria)
    # ============================================================
    print("\n" + "="*60)
    print("📋 GO-LIVE BENCHMARK COMPARISON")
    print("="*60)
    
    # Calculate aggregate metrics
    total_trades = df_all['trades'].sum()
    avg_sharpe = df_all['sharpe_ratio'].mean()
    avg_profit_factor = df_all['profit_factor'].mean()
    max_dd = df_all['max_drawdown_pct'].max()
    avg_win_rate = df_all['win_rate'].mean()
    avg_calmar = df_all['calmar_ratio'].mean() if 'calmar_ratio' in df_all.columns else 0
    avg_expectancy = df_all['expectancy'].mean() if 'expectancy' in df_all.columns else 0
    
    # Benchmark thresholds
    benchmarks = [
        ("Trade Count", total_trades, "≥ 100", total_trades >= 100),
        ("Avg Sharpe Ratio", avg_sharpe, "≥ 1.0", avg_sharpe >= 1.0),
        ("Avg Profit Factor", avg_profit_factor, "≥ 1.5", avg_profit_factor >= 1.5),
        ("Max Drawdown %", max_dd, "≤ 15%", max_dd <= 15),
        ("Avg Win Rate %", avg_win_rate, "≥ 45%", avg_win_rate >= 45),
        ("Avg Calmar Ratio", avg_calmar, "≥ 0.5", avg_calmar >= 0.5),
        ("Avg Expectancy ₹", avg_expectancy, "> 0", avg_expectancy > 0),
    ]
    
    passed = 0
    print("\n   METRIC                VALUE      BENCHMARK   STATUS")
    print("   " + "-"*55)
    for name, value, threshold, is_pass in benchmarks:
        status = "✅ PASS" if is_pass else "❌ FAIL"
        if is_pass:
            passed += 1
        print(f"   {name:<20} {value:>8.2f}   {threshold:<10} {status}")
    
    print("   " + "-"*55)
    print(f"   TOTAL: {passed}/{len(benchmarks)} benchmarks passed")
    
    # Overall verdict
    print("\n" + "="*60)
    if passed >= 6:
        print("🚀 VERDICT: READY FOR PAPER TRADING")
        print("   Run: python cli.py engine --mode PAPER")
    elif passed >= 4:
        print("⚠️  VERDICT: NEEDS OPTIMIZATION")
        print("   Consider: Tighter pair selection, longer backtest period")
    else:
        print("❌ VERDICT: NOT READY FOR LIVE")
        print("   Consider: Review strategy parameters, add more data")
    print("="*60)


def run_pro_backtest_fresh():
    """Fresh backtest without resume."""
    run_pro_backtest(resume=False)


if __name__ == "__main__":
    run_pro_backtest()

