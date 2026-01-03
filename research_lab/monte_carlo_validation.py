"""
Monte Carlo Parameter Sensitivity Testing

Research-backed robustness testing for pair trading strategies.
Per research: Robust strategies perform well across parameter ranges.

Usage:
    from research_lab.monte_carlo_validation import run_monte_carlo_sensitivity
    results = run_monte_carlo_sensitivity(pair_data)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from itertools import product
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_monte_carlo_sensitivity(
    pair_data: Dict,
    backtest_engine,
    z_entry_range: List[float] = None,
    z_exit_range: List[float] = None,
    lookback_range: List[int] = None,
    max_hold_range: List[int] = None,
    n_samples: int = 50
) -> Dict:
    """
    Run Monte Carlo sensitivity analysis on pair trading parameters.
    
    Tests strategy robustness by varying parameters and measuring
    outcome distribution per research best practices.
    
    Args:
        pair_data: Pair configuration dict
        backtest_engine: HybridBacktest instance
        z_entry_range: Z-score entry thresholds to test
        z_exit_range: Z-score exit thresholds to test
        lookback_range: Lookback periods to test
        max_hold_range: Max holding days to test
        n_samples: Number of random parameter samples
        
    Returns:
        Dict with sensitivity analysis results
    """
    # Default parameter ranges per research
    if z_entry_range is None:
        z_entry_range = [1.8, 2.0, 2.2, 2.5, 2.8, 3.0]
    if z_exit_range is None:
        z_exit_range = [0.3, 0.5, 0.8, 1.0]
    if lookback_range is None:
        lookback_range = [150, 200, 250, 300]
    if max_hold_range is None:
        max_hold_range = [5, 10, 15, 20]
    
    # Generate parameter combinations
    all_combinations = list(product(z_entry_range, z_exit_range, lookback_range, max_hold_range))
    
    # Sample if too many combinations
    if len(all_combinations) > n_samples:
        np.random.seed(42)
        indices = np.random.choice(len(all_combinations), n_samples, replace=False)
        combinations = [all_combinations[i] for i in indices]
    else:
        combinations = all_combinations
    
    # Run backtests for each combination
    results = []
    for z_entry, z_exit, lookback, max_hold in combinations:
        # Skip invalid combinations
        if z_exit >= z_entry:
            continue
        
        # Create modified pair data
        test_params = pair_data.copy()
        test_params['z_entry'] = z_entry
        test_params['z_exit'] = z_exit
        test_params['lookback'] = lookback
        test_params['max_hold'] = max_hold
        
        try:
            # Run backtest with modified parameters
            # Note: This requires modifying the backtest to accept these params
            result = backtest_engine.run(test_params)
            
            if 'error' not in result:
                results.append({
                    'z_entry': z_entry,
                    'z_exit': z_exit,
                    'lookback': lookback,
                    'max_hold': max_hold,
                    'return_pct': result.get('return_pct', 0),
                    'sharpe': result.get('sharpe_ratio', 0),
                    'win_rate': result.get('win_rate', 0),
                    'trades': result.get('trades', 0)
                })
        except Exception:
            continue
    
    if not results:
        return {'error': 'No valid Monte Carlo results', 'pair': pair_data.get('pair', 'UNKNOWN')}
    
    # Analyze results distribution
    df_results = pd.DataFrame(results)
    
    return {
        'pair': pair_data.get('leg1', '') + '-' + pair_data.get('leg2', ''),
        'n_simulations': len(results),
        'return': {
            'mean': round(df_results['return_pct'].mean(), 2),
            'std': round(df_results['return_pct'].std(), 2),
            'min': round(df_results['return_pct'].min(), 2),
            'max': round(df_results['return_pct'].max(), 2),
            'pct_positive': round((df_results['return_pct'] > 0).mean() * 100, 1)
        },
        'sharpe': {
            'mean': round(df_results['sharpe'].mean(), 2),
            'std': round(df_results['sharpe'].std(), 2)
        },
        'win_rate': {
            'mean': round(df_results['win_rate'].mean(), 1),
            'std': round(df_results['win_rate'].std(), 1)
        },
        'robustness_score': _calculate_robustness_score(df_results),
        'best_params': _get_best_params(df_results),
        'parameter_sensitivity': _analyze_parameter_sensitivity(df_results)
    }


def _calculate_robustness_score(df: pd.DataFrame) -> float:
    """
    Calculate robustness score (0-100).
    
    High score = consistent performance across parameter ranges.
    Low score = performance depends heavily on specific parameters.
    """
    if len(df) < 5:
        return 0.0
    
    # Factors contributing to robustness:
    # 1. Percentage of profitable configurations
    pct_profitable = (df['return_pct'] > 0).mean()
    
    # 2. Low return variance (consistent)
    return_cv = df['return_pct'].std() / (abs(df['return_pct'].mean()) + 0.001)
    consistency_score = max(0, 1 - return_cv)  # Lower CV = higher score
    
    # 3. Positive average Sharpe
    sharpe_score = min(1, max(0, df['sharpe'].mean() / 2))  # Sharpe 2+ = perfect
    
    # Weighted combination
    robustness = (pct_profitable * 40 + consistency_score * 30 + sharpe_score * 30)
    return round(robustness, 1)


def _get_best_params(df: pd.DataFrame) -> Dict:
    """Get parameters with highest Sharpe ratio."""
    if len(df) == 0:
        return {}
    
    best_idx = df['sharpe'].idxmax()
    best_row = df.loc[best_idx]
    
    return {
        'z_entry': best_row['z_entry'],
        'z_exit': best_row['z_exit'],
        'lookback': int(best_row['lookback']),
        'max_hold': int(best_row['max_hold']),
        'return_pct': best_row['return_pct'],
        'sharpe': best_row['sharpe']
    }


def _analyze_parameter_sensitivity(df: pd.DataFrame) -> Dict:
    """Analyze which parameters have most impact on returns."""
    if len(df) < 10:
        return {}
    
    sensitivity = {}
    
    for param in ['z_entry', 'z_exit', 'lookback', 'max_hold']:
        grouped = df.groupby(param)['return_pct'].mean()
        if len(grouped) > 1:
            sensitivity[param] = {
                'range': round(grouped.max() - grouped.min(), 2),
                'best_value': grouped.idxmax()
            }
    
    return sensitivity


if __name__ == "__main__":
    print("Monte Carlo Sensitivity Testing Module")
    print("Import and use run_monte_carlo_sensitivity() function")
