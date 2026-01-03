"""
Pair Strategy - Integrated with Unified Core Module

Uses the new core/ module for signal generation while preserving
the AssumptionGuardian for health monitoring.
"""

import pandas as pd
import numpy as np
from strategies.guardian import AssumptionGuardian

# Import from new core module
from core import (
    calculate_live_z_score_from_params,
    generate_signal,
    perform_regression,
    calculate_rolling_statistics,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    STOP_LOSS_THRESHOLD
)


class PairStrategy:
    """
    Professional Strategy Wrapper with Unified Core Integration.
    
    Now uses core/ module for:
    - Z-score calculation (regression-based)
    - Signal generation (entry/exit/stop)
    
    Still uses AssumptionGuardian for:
    - Health monitoring
    - Assumption checking
    """
    
    def __init__(self, hedge_ratio: float, intercept: float):
        """
        Initialize strategy with hedge ratio and intercept.
        
        Args:
            hedge_ratio: Beta from regression (Y = β*X + c)
            intercept: Regression intercept
        """
        # Core parameters
        self.beta = hedge_ratio
        self.intercept = intercept
        
        # Rolling statistics (updated during calibration)
        self.residual_std_dev = None
        self.residual_mean = 0.0
        self._calibrated = False
        
        # Signal thresholds (from core constants)
        self.entry_z = ENTRY_THRESHOLD  # 2.5
        self.exit_z = EXIT_THRESHOLD    # 1.0
        self.stop_z = STOP_LOSS_THRESHOLD  # 3.0
        
        # The Guardian (Health Monitor) - preserved from original
        self.guardian = AssumptionGuardian(lookback_window=60)
        self.guardian.calibrate(hedge_ratio)
    
    def calibrate(self, series_y: pd.Series, series_x: pd.Series):
        """
        Calibrate strategy with historical data.
        
        Computes residual statistics for z-score calculation.
        
        Args:
            series_y: Historical Y (dependent) prices
            series_x: Historical X (independent) prices
        """
        prices_y = series_y.values
        prices_x = series_x.values
        
        # Calculate residuals using stored beta/intercept
        predicted = self.intercept + (self.beta * prices_x)
        residuals = prices_y - predicted
        
        # Get rolling statistics (last 20 days for z-score)
        rolling_mean, rolling_std = calculate_rolling_statistics(residuals, lookback=20)
        
        self.residual_mean = rolling_mean[-1] if len(rolling_mean) > 0 else 0.0
        self.residual_std_dev = rolling_std[-1] if len(rolling_std) > 0 else np.std(residuals)
        self._calibrated = True
    
    def generate_signal(self, input_y, input_x) -> dict:
        """
        Generate trading signal using new core module.
        
        1. Clean Data (DataFrame -> Series)
        2. Calibrate if needed
        3. Check Health (Guardian)
        4. Calculate Z-Score (core module)
        5. Generate Signal (core module)
        
        Args:
            input_y: Y stock price data (Series or DataFrame)
            input_x: X stock price data (Series or DataFrame)
            
        Returns:
            Dict with signal, zscore, health, and health_reason
        """
        # --- Clean inputs to Series ---
        s_y = self._to_series(input_y)
        s_x = self._to_series(input_x)
        
        # Align Data
        df = pd.concat([s_y, s_x], axis=1).dropna()
        if len(df) < 20:
            return {'signal': 'WAIT', 'health': 'YELLOW', 'zscore': 0.0}
        
        clean_y = df.iloc[:, 0]
        clean_x = df.iloc[:, 1]
        
        # Latest Prices
        latest_y = float(clean_y.iloc[-1])
        latest_x = float(clean_x.iloc[-1])
        
        # Safety Filter
        if latest_y <= 0 or latest_x <= 0 or pd.isna(latest_y) or pd.isna(latest_x):
            return {'signal': 'WAIT', 'health': 'YELLOW', 'zscore': 0.0}
        
        # Z-score calculation uses latest_y and latest_x
        
        # --- Calibrate on first run ---
        if not self._calibrated:
            self.calibrate(clean_y, clean_x)
        
        # --- Guardian Health Check (preserved) ---
        self.guardian.update_data(latest_y, latest_x)
        status, reason = self.guardian.diagnose()
        
        if status == "RED":
            return {
                'signal': 'STOP_LOSS',
                'reason': f"SYSTEM HALT: {reason}",
                'health': status,
                'health_reason': reason,
                'zscore': 0.0
            }
        
        # --- Core Module: Z-Score Calculation ---
        live_data = calculate_live_z_score_from_params(
            price_x=latest_x,
            price_y=latest_y,
            intercept=self.intercept,
            beta=self.beta,
            residual_std_dev=self.residual_std_dev if self.residual_std_dev else 1.0
        )
        
        current_z = live_data['z_score']
        
        # --- Core Module: Signal Generation ---
        # Determine current position status (assume NONE for entry signal)
        core_signal = generate_signal(current_z, current_position="NONE")
        
        # Map core signal to legacy format
        signal = self._map_signal(core_signal, current_z)
        
        return {
            'signal': signal,
            'zscore': round(current_z, 2),
            'health': status,
            'health_reason': reason,
            'residual': live_data['residual'],
            'predicted_y': live_data['predicted_y']
        }
    
    def generate_exit_signal(self, current_z: float, position_side: str) -> dict:
        """
        Generate exit signal for an existing position.
        
        Args:
            current_z: Current z-score
            position_side: "LONG" or "SHORT"
            
        Returns:
            Dict with exit signal info
        """
        core_signal = generate_signal(current_z, current_position=position_side)
        
        return {
            'action': core_signal['action'],
            'type': core_signal['type'],
            'reason': core_signal['reason'],
            'zscore': current_z
        }
    
    def _to_series(self, data) -> pd.Series:
        """Convert DataFrame or Series to Series."""
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                return data['close']
            else:
                return data.iloc[:, 0]
        return data
    
    def _map_signal(self, core_signal: dict, z: float) -> str:
        """
        Map core module signal to legacy format.
        
        Legacy signals: LONG_SPREAD, SHORT_SPREAD, EXIT, STOP_LOSS, WAIT
        """
        action = core_signal['action']
        sig_type = core_signal['type']
        
        if action == 'ENTER':
            if sig_type == 'LONG':
                return 'LONG_SPREAD'
            elif sig_type == 'SHORT':
                return 'SHORT_SPREAD'
        elif action == 'EXIT':
            if sig_type == 'STOP_LOSS':
                return 'STOP_LOSS'
            else:
                return 'EXIT'
        
        # HOLD or no signal
        if abs(z) < 1.0:
            return 'EXIT'  # Near mean (PDF: Exit at ±1.0 SD)
        
        return 'WAIT'


# Backward compatibility - import StatArbBot for legacy code
try:
    from strategies.stat_arb_bot import StatArbBot
except ImportError:
    StatArbBot = None
