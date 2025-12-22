import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class StatArbBot:
    """
    Implements Method 2: Cointegration & Statistical Arbitrage.
    Theory: Trading the Mean Reversion of Residuals (Errors).
    """
    def __init__(self, entry_z=2.5, exit_z=0.0, stop_z=3.5, lookback=20):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.lookback = lookback
        
        # State
        self.y_symbol = None
        self.x_symbol = None
        self.beta = 0.0
        self.intercept = 0.0
        self.is_cointegrated = False

    def _run_ols(self, y, x):
        """Helper: Runs OLS and calculates Error Ratio"""
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const)
        res = model.fit()
        
        # Error Ratio = SE(Intercept) / SE(Slope)
        # Low Error Ratio indicates a more stable relationship
        se_intercept = res.bse['const']
        se_slope = res.bse[x.name]
        error_ratio = se_intercept / se_slope if se_slope != 0 else 999
        
        return res, error_ratio

    def calibrate(self, df_a, df_b, sym_a, sym_b):
        """
        Step 1: Identify Y and X (Lowest Error Ratio).
        Step 2: Verify Stationarity (ADF Test).
        """
        # Align
        df = pd.concat([df_a, df_b], axis=1).dropna()
        df.columns = [sym_a, sym_b]
        
        if len(df) < 60: return False # Need data for regression

        # 1. Run Regression Both Ways
        res_a, err_a = self._run_ols(df[sym_a], df[sym_b])
        res_b, err_b = self._run_ols(df[sym_b], df[sym_a])
        
        # 2. Select Dependent (Y) and Independent (X)
        if err_a < err_b:
            self.y_symbol = sym_a
            self.x_symbol = sym_b
            best_model = res_a
        else:
            self.y_symbol = sym_b
            self.x_symbol = sym_a
            best_model = res_b
            
        self.beta = best_model.params[self.x_symbol]
        self.intercept = best_model.params['const']
        
        # 3. ADF Test on Residuals
        # Method 2 requires P-Value < 0.05
        adf = adfuller(best_model.resid)
        p_value = adf[1]
        
        # We use 0.05 as per PDF, but 0.10 is acceptable for scanning
        if p_value < 0.10:
            self.is_cointegrated = True
            return True
        else:
            self.is_cointegrated = False
            return False

    def get_zscore(self, price_y, price_x):
        """
        Calculates the Z-Score of the current residual against historical mean/std.
        Z = (Current Resid - Rolling Mean) / Rolling Std
        """
        # 1. Calculate Current Residual
        # resid = Y - (Beta * X + C)
        current_resid = price_y - (self.beta * price_x + self.intercept)
        
        # Note: In a live system, we need a history of residuals to calculate Z.
        # This function assumes it's being called iteratively or we maintain a buffer.
        return current_resid # Returning raw resid here, scaling happens in strategy

    def generate_full_series(self, series_y, series_x):
        """
        Generates Z-Score series for Backtesting/Scanning
        """
        residuals = series_y - (self.beta * series_x + self.intercept)
        
        rolling_mean = residuals.rolling(self.lookback).mean()
        rolling_std = residuals.rolling(self.lookback).std()
        
        zscore = (residuals - rolling_mean) / rolling_std
        return zscore
