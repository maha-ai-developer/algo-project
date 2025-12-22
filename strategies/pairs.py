import pandas as pd
import numpy as np
from strategies.stat_arb_bot import StatArbBot
from strategies.guardian import AssumptionGuardian

class PairStrategy:
    """
    Professional Strategy Wrapper with 'Live Assumption Health Dashboard'.
    """
    def __init__(self, hedge_ratio, intercept):
        # The Brain (Math)
        self.bot = StatArbBot(entry_z=2.0, exit_z=0.0, stop_z=4.0)
        self.bot.beta = hedge_ratio
        self.bot.intercept = intercept
        
        # The Guardian (Health Monitor)
        self.guardian = AssumptionGuardian(lookback_window=60)
        self.guardian.calibrate(hedge_ratio)

    def generate_signal(self, input_y, input_x):
        """
        1. Clean Data (DataFrame -> Series)
        2. Check Health (Guardian)
        3. Calculate Z-Score
        """
        # --- FIX 1: Ensure Inputs are Series (Not DataFrames) ---
        # If input is a DataFrame (e.g. has columns like 'close'), strip it to a Series
        if isinstance(input_y, pd.DataFrame):
            # Try to get 'close', otherwise take the first column
            if 'close' in input_y.columns:
                s_y = input_y['close']
            else:
                s_y = input_y.iloc[:, 0]
        else:
            s_y = input_y

        if isinstance(input_x, pd.DataFrame):
            if 'close' in input_x.columns:
                s_x = input_x['close']
            else:
                s_x = input_x.iloc[:, 0]
        else:
            s_x = input_x

        # Align Data
        df = pd.concat([s_y, s_x], axis=1).dropna()
        if len(df) < 20: 
            return {'signal': 'WAIT', 'health': 'YELLOW', 'zscore': 0.0}
        
        # Separate back into clean Series
        clean_y = df.iloc[:, 0]
        clean_x = df.iloc[:, 1]

        # Latest Prices (Scalars)
        latest_y = clean_y.iloc[-1]
        latest_x = clean_x.iloc[-1]
        
        # ✅ SAFETY FILTER: Ignore Zeros or NaNs
        if latest_y <= 0 or latest_x <= 0 or pd.isna(latest_y) or pd.isna(latest_x):
            # If bad data, return previous state but DO NOT update Guardian memory
            return {'signal': 'WAIT', 'health': 'YELLOW', 'zscore': 0.0}

        # 1. Feed Data to Guardian
        self.guardian.update_data(latest_y, latest_x)
        
        # 2. Get Health Diagnosis
        status, reason = self.guardian.diagnose()
        
        # 3. Act on Health Status
        if status == "RED":
            return {
                'signal': 'STOP_LOSS',
                'reason': f"SYSTEM HALT: {reason}",
                'health': status,
                'health_reason': reason,
                'zscore': 0.0
            }
            
        # 4. Run Math (Only if healthy)
        z_series = self.bot.generate_full_series(clean_y, clean_x)
        
        # --- FIX 2: Ensure Z-Score is a scalar Float ---
        if z_series.empty:
            return {'signal': 'WAIT', 'health': status, 'zscore': 0.0}
            
        current_z = float(z_series.iloc[-1]) # <--- FORCE FLOAT
        
        signal = "WAIT"
        
        # Logic
        if current_z < -self.bot.entry_z and current_z > -self.bot.stop_z:
            signal = "LONG_SPREAD"
        elif current_z > self.bot.entry_z and current_z < self.bot.stop_z:
            signal = "SHORT_SPREAD"
        elif abs(current_z) < 0.5:
            signal = "EXIT"
        elif abs(current_z) > self.bot.stop_z:
            signal = "STOP_LOSS" # Z-Score Stop Loss
            
        return {
            'signal': signal,
            'zscore': round(current_z, 2),
            'health': status,
            'health_reason': reason
        }
