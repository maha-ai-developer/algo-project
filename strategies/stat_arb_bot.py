import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

class StatArbBot:
    """
    Implements Method 2: Cointegration & Statistical Arbitrage.
    Theory: Trading the Mean Reversion of Residuals (Errors).
    """
    def __init__(self, entry_z=2.5, exit_z=0.0, stop_z=3.0, lookback=20):
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
        
        # Half-life and Hurst (NEW - Checklist Enhancement)
        self.half_life = np.inf
        self.hurst_exponent = 0.5
        self.is_valid_halflife = False
        self.is_mean_reverting = False
        self.adf_pvalue = 1.0
        self.adf_statistic = 0.0
        self.kpss_pvalue = 0.0  # KPSS for dual stationarity check

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

    def calculate_half_life(self, residuals: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion using Ornstein-Uhlenbeck process.
        
        Theory: residual_t - residual_{t-1} = -theta * residual_{t-1} + epsilon
        Half-life = ln(2) / theta
        
        Args:
            residuals: Residual series from cointegration regression
            
        Returns:
            Half-life in trading days. Returns np.inf if non-mean-reverting.
        """
        if len(residuals) < 20:
            return np.inf
        
        try:
            # Reset index for proper alignment
            resid = residuals.reset_index(drop=True)
            
            # Lagged residuals (y_t-1)
            resid_lag = resid.shift(1)
            
            # Delta residuals (y_t - y_t-1)
            resid_delta = resid.diff()
            
            # Remove NaN from shift/diff
            resid_lag = resid_lag.iloc[1:]
            resid_delta = resid_delta.iloc[1:]
            
            if len(resid_lag) < 10:
                return np.inf
            
            # Regress: delta_residual = alpha + theta * residual_lag + epsilon
            x_const = sm.add_constant(resid_lag)
            model = sm.OLS(resid_delta, x_const).fit()
            
            # theta is the coefficient on the lagged residual
            theta = model.params.iloc[1]
            
            # For mean reversion, theta should be negative
            # Half-life = ln(2) / |theta|
            if theta >= 0:
                # Positive theta means diverging, not mean-reverting
                return np.inf
            
            half_life = -np.log(2) / theta
            return max(0.1, min(half_life, 1000))  # Clamp to reasonable range
            
        except Exception:
            return np.inf

    def calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent using simplified R/S analysis.
        
        H < 0.5: Mean-reverting (good for pair trading)
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Args:
            series: Time series data
            max_lag: Maximum lag for analysis
            
        Returns:
            Hurst exponent (0 to 1)
        """
        if len(series) < max_lag * 2:
            return 0.5  # Default to random walk if insufficient data
        
        try:
            lags = range(2, min(max_lag, len(series) // 4))
            
            # Variance ratio method (simplified)
            tau = []
            rs_values = []
            
            for lag in lags:
                tau.append(lag)
                # Calculate standard deviation of different lags
                std_lag = series.diff(lag).std()
                std_1 = series.diff().std()
                if std_1 > 0:
                    rs_values.append(std_lag / (std_1 * np.sqrt(lag)))
            
            if len(tau) < 3 or len(rs_values) < 3:
                return 0.5
            
            # Fit log-log regression: log(R/S) = H * log(lag)
            log_tau = np.log(tau)
            log_rs = np.log(rs_values)
            
            # Linear regression
            slope, _ = np.polyfit(log_tau, log_rs, 1)
            
            # Hurst exponent
            hurst = slope + 0.5
            return max(0.0, min(1.0, hurst))  # Clamp to [0, 1]
            
        except Exception:
            return 0.5

    def calibrate(self, df_a, df_b, sym_a, sym_b):
        """
        Step 1: Identify Y and X (Lowest Error Ratio).
        Step 2: Verify Stationarity (ADF Test).
        Step 3: Calculate Half-Life and Hurst Exponent (NEW).
        
        Returns:
            dict with calibration results, or False if insufficient data
        """
        # Align
        df = pd.concat([df_a, df_b], axis=1).dropna()
        df.columns = [sym_a, sym_b]
        
        if len(df) < 60: 
            return False  # Need data for regression

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
        residuals = best_model.resid
        adf = adfuller(residuals, maxlag=None, autolag='AIC')
        adf_pvalue = adf[1]
        
        # Store sigma (Standard Error of Residuals) - crucial for Z-score calculation
        self.sigma = np.std(residuals)
        
        # Store ADF details for reporting
        self.adf_pvalue = adf_pvalue
        self.adf_statistic = adf[0]
        
        # SIMPLIFIED validation: ADF p-value < 0.05 only (per user spec)
        self.is_cointegrated = adf_pvalue < 0.05
        
        # Return result if cointegrated
        if self.is_cointegrated:
            return {
                'is_cointegrated': True,
                'adf_pvalue': round(adf_pvalue, 4),
                'beta': self.beta,
                'intercept': self.intercept,
                'sigma': round(self.sigma, 4),
                'y_symbol': self.y_symbol,
                'x_symbol': self.x_symbol
            }
        else:
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

    def print_regression_stats(self, series_y, series_x, sym_y, sym_x):
        """
        Prints detailed regression statistics including:
        - Regression Statistics (R², Adjusted R², Std Error)
        - ANOVA Table
        - Coefficients Table
        - Residual Output (first 8 observations)
        """
        # Align data
        df = pd.concat([series_y, series_x], axis=1).dropna()
        df.columns = [sym_y, sym_x]
        
        if len(df) < 20:
            print("   ⚠️ Insufficient data for regression statistics")
            return
        
        # Run OLS
        x_const = sm.add_constant(df[sym_x])
        model = sm.OLS(df[sym_y], x_const)
        res = model.fit()
        
        # ═══════════════════════════════════════════════════════════
        print(f"\n   {'═'*70}")
        print(f"   📊 REGRESSION ANALYSIS: {sym_y} = β × {sym_x} + Intercept")
        print(f"   {'═'*70}")
        
        # Regression Statistics Table
        print(f"\n   ┌{'─'*40}┬{'─'*20}┐")
        print(f"   │ {'Metric':<38} │ {'Value':>18} │")
        print(f"   ├{'─'*40}┼{'─'*20}┤")
        print(f"   │ {'Multiple R':<38} │ {res.rsquared**0.5:>18.6f} │")
        print(f"   │ {'R Square':<38} │ {res.rsquared:>18.6f} │")
        print(f"   │ {'Adjusted R Square':<38} │ {res.rsquared_adj:>18.6f} │")
        print(f"   │ {'Standard Error':<38} │ {res.mse_resid**0.5:>18.4f} │")
        print(f"   │ {'Observations':<38} │ {int(res.nobs):>18} │")
        print(f"   └{'─'*40}┴{'─'*20}┘")
        
        # ANOVA Table
        print(f"\n   ┌{'─'*12}┬{'─'*5}┬{'─'*18}┬{'─'*18}┬{'─'*12}┬{'─'*14}┐")
        print(f"   │ {'Source':<10} │ {'df':>3} │ {'SS':>16} │ {'MS':>16} │ {'F':>10} │ {'Sig F':>12} │")
        print(f"   ├{'─'*12}┼{'─'*5}┼{'─'*18}┼{'─'*18}┼{'─'*12}┼{'─'*14}┤")
        
        df_m = res.df_model
        df_r = res.df_resid
        ss_reg = res.ess
        ss_res = res.ssr
        ms_reg = ss_reg / df_m if df_m > 0 else 0
        ms_res = ss_res / df_r if df_r > 0 else 0
        
        print(f"   │ {'Regression':<10} │ {int(df_m):>3} │ {ss_reg:>16.2f} │ {ms_reg:>16.2f} │ {res.fvalue:>10.2f} │ {res.f_pvalue:>12.2e} │")
        print(f"   │ {'Residual':<10} │ {int(df_r):>3} │ {ss_res:>16.2f} │ {ms_res:>16.2f} │ {'':>10} │ {'':>12} │")
        print(f"   │ {'Total':<10} │ {int(df_m + df_r):>3} │ {ss_reg + ss_res:>16.2f} │ {'':>16} │ {'':>10} │ {'':>12} │")
        print(f"   └{'─'*12}┴{'─'*5}┴{'─'*18}┴{'─'*18}┴{'─'*12}┴{'─'*14}┘")
        
        # Coefficients Table
        print(f"\n   ┌{'─'*14}┬{'─'*14}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┐")
        print(f"   │ {'Variable':<12} │ {'Coefficient':>12} │ {'Std Error':>10} │ {'t Stat':>10} │ {'P-value':>10} │ {'Lower 95%':>10} │ {'Upper 95%':>10} │")
        print(f"   ├{'─'*14}┼{'─'*14}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┤")
        
        # Intercept
        conf = res.conf_int()
        print(f"   │ {'Intercept':<12} │ {res.params['const']:>12.4f} │ {res.bse['const']:>10.4f} │ {res.tvalues['const']:>10.4f} │ {res.pvalues['const']:>10.2e} │ {conf.loc['const', 0]:>10.2f} │ {conf.loc['const', 1]:>10.2f} │")
        
        # Beta (X variable)
        print(f"   │ {sym_x:<12} │ {res.params[sym_x]:>12.4f} │ {res.bse[sym_x]:>10.4f} │ {res.tvalues[sym_x]:>10.4f} │ {res.pvalues[sym_x]:>10.2e} │ {conf.loc[sym_x, 0]:>10.2f} │ {conf.loc[sym_x, 1]:>10.2f} │")
        print(f"   └{'─'*14}┴{'─'*14}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┘")
        
        # Residual Output (first 8)
        print(f"\n   ┌{'─'*8}┬{'─'*18}┬{'─'*18}┐")
        print(f"   │ {'Obs':>6} │ {'Predicted ' + sym_y:>16} │ {'Residual':>16} │")
        print(f"   ├{'─'*8}┼{'─'*18}┼{'─'*18}┤")
        
        predicted = res.fittedvalues
        residuals = res.resid
        
        for i in range(min(8, len(residuals))):
            print(f"   │ {i+1:>6} │ {predicted.iloc[i]:>16.2f} │ {residuals.iloc[i]:>16.2f} │")
        
        if len(residuals) > 8:
            print(f"   │ {'...':>6} │ {'...':>16} │ {'...':>16} │")
        
        print(f"   └{'─'*8}┴{'─'*18}┴{'─'*18}┘")
        
        # ADF Test on Residuals
        adf = adfuller(residuals)
        print(f"\n   📈 ADF Test on Residuals:")
        print(f"      Statistic: {adf[0]:.4f} | P-value: {adf[1]:.4f} | {'✓ STATIONARY' if adf[1] < 0.05 else '✗ NON-STATIONARY'}")
        print(f"   {'═'*70}\n")
