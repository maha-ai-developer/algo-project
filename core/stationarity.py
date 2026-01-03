"""
Unified Pair Trading System - ADF Stationarity Test (Module 4)

Tests whether residuals from pair regression are stationary (mean-reverting).
Uses the Augmented Dickey-Fuller (ADF) test.

Stationarity is REQUIRED for pair trading:
- p-value ≤ 0.05: Residuals are stationary (TRADABLE)
- p-value > 0.05: Residuals are non-stationary (NOT TRADABLE)
"""

import numpy as np
from typing import Union, List, Dict, Tuple
from .constants import ADF_THRESHOLD, ADF_EXCELLENT


def perform_adf_test(residuals: Union[np.ndarray, List[float]]) -> Dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Algorithm from unified_pair_trading.txt Module 4:
    1. Create first differences: Δy_t = y_t - y_{t-1}
    2. Create lags: y_{t-1}
    3. Regress: Δy_t = α + β·y_{t-1} + ε
    4. Calculate test statistic = β / SE(β)
    5. Compare to critical values for p-value
    
    Critical values (approximate):
    - test_stat < -3.43: p ≈ 0.01 (highly stationary)
    - test_stat < -2.86: p ≈ 0.05 (stationary)
    - test_stat < -2.57: p ≈ 0.10 (marginally stationary)
    - else: p ≈ 0.50 (not stationary)
    
    Args:
        residuals: Array of residual values from regression
        
    Returns:
        Dictionary with:
        - p_value: Approximate p-value
        - is_stationary: Boolean (p_value ≤ 0.05)
        - test_statistic: ADF test statistic
        - critical_values: Dict of critical values
    """
    residuals = np.array(residuals, dtype=np.float64)
    n = len(residuals)
    
    if n < 10:
        return {
            'p_value': 1.0,
            'is_stationary': False,
            'test_statistic': 0.0,
            'critical_values': {},
            'error': 'Insufficient data points'
        }
    
    # Step 1: Create first differences
    # Δy_t = y_t - y_{t-1}
    differences = residuals[1:] - residuals[:-1]
    
    # Step 2: Create lags
    # y_{t-1}
    lags = residuals[:-1]
    
    # Step 3: Simple regression to get test statistic
    # We're testing if β < 0 (mean-reverting)
    n_obs = len(differences)
    
    # Add intercept for regression
    # Δy = α + β·y_{t-1}
    X = np.column_stack([np.ones(n_obs), lags])
    y = differences
    
    try:
        # OLS: β = (X'X)^(-1) X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        
        # Residuals from this regression
        y_pred = X @ beta
        reg_residuals = y - y_pred
        
        # Standard error of regression
        sse = np.sum(reg_residuals ** 2)
        se_regression = np.sqrt(sse / (n_obs - 2))
        
        # Standard error of β (coefficient on lag)
        se_beta = se_regression * np.sqrt(XtX_inv[1, 1])
        
        # Test statistic
        test_stat = beta[1] / se_beta if se_beta > 0 else 0.0
        
    except np.linalg.LinAlgError:
        return {
            'p_value': 1.0,
            'is_stationary': False,
            'test_statistic': 0.0,
            'critical_values': {},
            'error': 'Matrix inversion failed'
        }
    
    # Step 4: Determine p-value from test statistic
    # Using approximate critical values for no trend, no constant case
    # (simplified - real ADF uses more complex tables)
    p_value = _approximate_p_value(test_stat, n_obs)
    
    # Critical values
    critical_values = {
        '1%': -3.43,
        '5%': -2.86,
        '10%': -2.57
    }
    
    return {
        'p_value': p_value,
        'is_stationary': p_value <= ADF_THRESHOLD,
        'test_statistic': float(test_stat),
        'critical_values': critical_values
    }


def _approximate_p_value(test_stat: float, n: int) -> float:
    """
    Approximate p-value from ADF test statistic.
    
    Uses simplified critical value mapping.
    For production, consider using statsmodels.tsa.stattools.adfuller
    
    Args:
        test_stat: ADF test statistic
        n: Number of observations
        
    Returns:
        Approximate p-value
    """
    # Approximate critical values (for n > 100)
    # These are for the case with constant, no trend
    if test_stat < -3.43:
        return 0.01
    elif test_stat < -2.86:
        return 0.05
    elif test_stat < -2.57:
        return 0.10
    elif test_stat < -1.94:
        return 0.25
    else:
        return 0.50


def perform_adf_test_statsmodels(
    residuals: Union[np.ndarray, List[float]],
    max_lags: int = None
) -> Dict:
    """
    Perform ADF test using statsmodels (more accurate).
    
    Falls back to simple implementation if statsmodels not available.
    
    Args:
        residuals: Array of residual values
        max_lags: Maximum lags to include (auto if None)
        
    Returns:
        Dictionary with ADF test results
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        
        residuals = np.array(residuals, dtype=np.float64)
        
        # Run ADF test
        result = adfuller(residuals, maxlag=max_lags, autolag='AIC')
        
        return {
            'p_value': float(result[1]),
            'is_stationary': result[1] <= ADF_THRESHOLD,
            'test_statistic': float(result[0]),
            'critical_values': result[4],
            'used_lag': result[2],
            'n_obs': result[3]
        }
        
    except ImportError:
        # Fall back to simple implementation
        return perform_adf_test(residuals)


def classify_stationarity(p_value: float) -> Tuple[str, int]:
    """
    Classify stationarity quality and assign score.
    
    From architecture spec:
    - p ≤ 0.01: EXCELLENT, score 25
    - p ≤ 0.05: GOOD, score 25
    - p ≤ 0.10: MARGINAL, score 15
    - p > 0.10: POOR, score 0
    
    Args:
        p_value: ADF test p-value
        
    Returns:
        Tuple of (quality_string, score)
    """
    from .constants import SCORE_ADF
    
    if p_value <= ADF_EXCELLENT:
        return "EXCELLENT", SCORE_ADF
    elif p_value <= ADF_THRESHOLD:
        return "GOOD", SCORE_ADF
    elif p_value <= 0.10:
        return "MARGINAL", 15
    else:
        return "POOR", 0


def calculate_hurst_exponent(series: Union[np.ndarray, List[float]], max_lag: int = 20) -> float:
    """
    Calculate the Hurst Exponent to determine mean-reverting behavior.
    
    H < 0.5: Mean Reverting (Good for StatArb)
    H = 0.5: Random Walk (Geometric Brownian Motion)
    H > 0.5: Trending (Momentum)
    
    Args:
        series: Time series data (prices or residuals)
        max_lag: Maximum lag for R/S calculation
        
    Returns:
        Hurst Exponent (float)
    """
    try:
        ts = np.array(series)
        if len(ts) < max_lag + 5:
            return 0.5
            
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        
        # Polyfit to finding slope (H)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
        
    except Exception:
        return 0.5

