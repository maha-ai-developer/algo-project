"""
Unified Pair Trading System - Core Module

A comprehensive, scalable statistical arbitrage system with:
- Linear regression analysis
- Error ratio optimization
- Cointegration testing (ADF)
- Complex position sizing
- Intercept risk assessment
- Real-time trade execution
- Portfolio management

Architecture: 5 Layers, 18 Modules
"""

# Layer 1: Data Models
from .models import (
    StockData,
    RegressionResult,
    PairAnalysis,
    PositionSizing,
    RiskAssessment,
    Trade,
    Portfolio
)

# Constants
from .constants import (
    LOOKBACK_PERIOD,
    ADF_THRESHOLD,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    STOP_LOSS_THRESHOLD,
    PROFILE_CONSERVATIVE,
    PROFILE_MODERATE,
    PROFILE_AGGRESSIVE,
    QUALITY_EXCELLENT,
    QUALITY_GOOD,
    QUALITY_FAIR,
    QUALITY_POOR,
    INTERCEPT_HIGH_RISK
)

# Layer 2: Core Analytics
from .regression import (
    perform_regression,
    calculate_residual,
    calculate_z_score,
    calculate_rolling_statistics
)

from .error_ratio import (
    calculate_error_ratio,
    calculate_optimal_direction,
    calculate_optimal_direction_from_prices
)

from .stationarity import (
    perform_adf_test,
    perform_adf_test_statsmodels,
    classify_stationarity,
    calculate_hurst_exponent
)

from .pair_analyzer import (
    analyze_pair,
    analyze_pair_from_prices,
    update_pair_z_score
)

from .screener import (
    screen_sector_pairs,
    screen_all_sectors,
    screen_pairs_from_symbols,
    screen_pairs_from_price_dict,
    rank_pairs,
    format_screening_report
)

# Layer 3: Advanced Validation
from .intercept_risk import (
    assess_intercept_risk,
    calculate_intercept_score,
    format_intercept_report
)

from .validator import (
    validate_pair_for_trading,
    validate_pair_simple,
    format_validation_report
)

from .decision_engine import (
    make_trade_decision,
    batch_decisions,
    filter_tradable,
    get_best_opportunities,
    format_decision_report
)

# Layer 4: Execution Support
from .signal_generator import (
    calculate_live_z_score,
    calculate_live_z_score_from_params,
    generate_signal,
    generate_signal_with_prices,
    get_trade_direction,
    format_signal_summary
)

__all__ = [
    # Data Models
    'StockData',
    'RegressionResult',
    'PairAnalysis',
    'PositionSizing',
    'RiskAssessment',
    'Trade',
    'Portfolio',
    # Constants
    'LOOKBACK_PERIOD',
    'ADF_THRESHOLD',
    'ENTRY_THRESHOLD',
    'EXIT_THRESHOLD',
    'STOP_LOSS_THRESHOLD',
    'PROFILE_CONSERVATIVE',
    'PROFILE_MODERATE',
    'PROFILE_AGGRESSIVE',
    'QUALITY_EXCELLENT',
    'QUALITY_GOOD',
    'QUALITY_FAIR',
    'QUALITY_POOR',
    'INTERCEPT_HIGH_RISK',
    # Regression
    'perform_regression',
    'calculate_residual',
    'calculate_z_score',
    'calculate_rolling_statistics',
    # Error Ratio
    'calculate_error_ratio',
    'calculate_optimal_direction',
    'calculate_optimal_direction_from_prices',
    # Stationarity
    'perform_adf_test',
    'perform_adf_test_statsmodels',
    'classify_stationarity',
    'calculate_hurst_exponent',
    # Pair Analyzer
    'analyze_pair',
    'analyze_pair_from_prices',
    'update_pair_z_score',
    # Screener
    'screen_sector_pairs',
    'screen_all_sectors',
    'screen_pairs_from_symbols',
    'screen_pairs_from_price_dict',
    'rank_pairs',
    'format_screening_report',
    # Intercept Risk
    'assess_intercept_risk',
    'calculate_intercept_score',
    'format_intercept_report',
    # Validator
    'validate_pair_for_trading',
    'validate_pair_simple',
    'format_validation_report',
    # Decision Engine
    'make_trade_decision',
    'batch_decisions',
    'filter_tradable',
    'get_best_opportunities',
    'format_decision_report',
    # Signal Generator
    'calculate_live_z_score',
    'calculate_live_z_score_from_params',
    'generate_signal',
    'generate_signal_with_prices',
    'get_trade_direction',
    'format_signal_summary',
]
