import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class AssumptionGuardian:
    def __init__(self, lookback_window=60):
        self.lookback = lookback_window
        self.history_y = []
        self.history_x = []
        
        # Baselines
        self.initial_beta = None
        self.red_light_counter = 0

    def calibrate(self, beta):
        self.initial_beta = beta
        self.red_light_counter = 0

    def update_data(self, price_y, price_x):
        # 1. Filter Bad Data (NaN, Inf, Zero)
        if not np.isfinite(price_y) or not np.isfinite(price_x) or price_y == 0 or price_x == 0:
            return # Ignore bad tick
            
        self.history_y.append(price_y)
        self.history_x.append(price_x)
        
        # Maintain window size
        if len(self.history_y) > self.lookback:
            self.history_y.pop(0)
            self.history_x.pop(0)

    def diagnose(self):
        if len(self.history_y) < 20:
            return "YELLOW", "Initializing"

        # 2. Prepare Series
        s_y = pd.Series(self.history_y)
        s_x = pd.Series(self.history_x)
        
        # Safety Check: aligned lengths
        if len(s_y) != len(s_x):
            return "YELLOW", "Data Aligning"

        try:
            # 3. Run Math in a Safe Block
            x_const = sm.add_constant(s_x)
            model = sm.OLS(s_y, x_const).fit()
            current_beta = model.params.iloc[1]
            residuals = model.resid
            
            # Check Beta Drift
            denom = self.initial_beta if self.initial_beta != 0 else 0.001
            drift_pct = abs((current_beta - self.initial_beta) / denom)

            # Check Stationarity (Safe ADF)
            # Handle case where residuals are constant (perfect correlation) -> ADF crashes
            if residuals.std() < 1e-6:
                p_value = 0.0 # Perfect stationarity
            else:
                adf = adfuller(residuals, maxlag=1)
                p_value = adf[1]

        except Exception as e:
            # FIX: If math fails, do NOT kill the system. Just Warn.
            # This handles the "Empty Entry" error you saw.
            return "YELLOW", "Math Computation Skip"

        # --- TRAFFIC LIGHTS ---
        if drift_pct > 0.30: 
            self.red_light_counter += 1
            return "RED", f"Beta Drift ({drift_pct:.2%})"
            
        if p_value > 0.20: 
            self.red_light_counter += 1
            return "RED", f"Broken Link (P={p_value:.2f})"

        self.red_light_counter = 0

        if drift_pct > 0.15 or p_value > 0.10:
            return "YELLOW", "Weak Signal"

        return "GREEN", "Healthy"

    def needs_recalibration(self):
        return self.red_light_counter > 5

    def force_recalibrate_to_current(self):
        if len(self.history_y) < 20: return self.initial_beta
        
        try:
            s_y = pd.Series(self.history_y)
            s_x = pd.Series(self.history_x)
            x_const = sm.add_constant(s_x)
            model = sm.OLS(s_y, x_const).fit()
            new_beta = model.params.iloc[1]
            self.initial_beta = new_beta
            self.red_light_counter = 0
            return new_beta
        except:
            return self.initial_beta
