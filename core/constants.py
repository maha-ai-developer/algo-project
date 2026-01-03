"""
Unified Pair Trading System - Configuration Constants

All thresholds and parameters from the architecture specification.
"""

# ═══════════════════════════════════════════════════════════════════════════
# DATA PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

LOOKBACK_PERIOD = 200           # Days of historical data
MIN_DATA_COVERAGE = 0.95        # 95% data completeness required

# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

ADF_THRESHOLD = 0.05            # P-value ≤ 0.05 per Zerodha Varsity PDF (5% significance)
ADF_EXCELLENT = 0.01            # For excellent/strict validation
HURST_THRESHOLD = 0.50          # Strict Mean Reversion (H < 0.5)
MIN_CROSSING_RATE = 12          # Minimum 12 mean crossings per year
ERROR_RATIO_EXCELLENT = 0.15    # Error ratio < 0.15 is excellent
ERROR_RATIO_GOOD = 0.25         # Error ratio < 0.25 is good
ERROR_RATIO_MAX = 0.40          # Error ratio < 0.40 is acceptable

# ═══════════════════════════════════════════════════════════════════════════
# TRADING SIGNALS
# ═══════════════════════════════════════════════════════════════════════════

ENTRY_THRESHOLD = 2.5           # Enter at ±2.5 SD (Zerodha Varsity Page 47)
EXIT_THRESHOLD = 1.0            # Exit at ±1.0 SD (mean reversion)
STOP_LOSS_THRESHOLD = 3.0       # Stop loss at ±3.0 SD

# ═══════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════

BETA_DEVIATION_ACCEPTABLE = 5.0     # ≤5% beta deviation acceptable
SPOT_ADJUSTMENT_THRESHOLD = 0.02    # >2% difference needs spot market
MAX_LOT_COMBINATIONS = 20           # Test up to 20 lot combinations

# ═══════════════════════════════════════════════════════════════════════════
# INTERCEPT RISK THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

INTERCEPT_LOW_RISK = 10         # <10% unexplained is low risk
INTERCEPT_MODERATE = 25         # <25% unexplained is moderate
INTERCEPT_ELEVATED = 50         # <50% unexplained is elevated
INTERCEPT_HIGH_RISK = 70        # ≥70% unexplained is high risk

# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

MAX_POSITION_SIZE_PERCENT = 20  # Max 20% capital per trade
MAX_OPEN_POSITIONS = 5          # Max 5 concurrent positions
CAPITAL_BUFFER = 0.1            # Keep 10% buffer

# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

MIN_WIN_RATE = 50.0             # Minimum 50% win rate
MIN_PROFIT_FACTOR = 1.5         # Minimum 1.5 profit factor
MIN_SHARPE_RATIO = 1.0          # Minimum 1.0 Sharpe ratio

# ═══════════════════════════════════════════════════════════════════════════
# SCORING SYSTEM (100 POINTS TOTAL)
# ═══════════════════════════════════════════════════════════════════════════

SCORE_ADF = 25                  # ADF test score
SCORE_ZSCORE = 20               # Z-score signal score
SCORE_INTERCEPT = 30            # Intercept risk score
SCORE_POSITION = 25             # Position sizing score
SCORE_MAX = 100                 # Maximum total score

# Score thresholds for recommendations
SCORE_EXCELLENT = 80            # ≥80% is excellent
SCORE_GOOD = 60                 # ≥60% is good
SCORE_MARGINAL = 40             # ≥40% is marginal

# ═══════════════════════════════════════════════════════════════════════════
# QUALITY CLASSIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

QUALITY_EXCELLENT = "EXCELLENT"
QUALITY_GOOD = "GOOD"
QUALITY_FAIR = "FAIR"
QUALITY_POOR = "POOR"

# ═══════════════════════════════════════════════════════════════════════════
# INTERCEPT RISK CLASSIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

RISK_LOW = "LOW"
RISK_MODERATE = "MODERATE"
RISK_ELEVATED = "ELEVATED"
RISK_HIGH = "HIGH"
RISK_VERY_HIGH = "VERY HIGH"

# ═══════════════════════════════════════════════════════════════════════════
# TRADE DIRECTIONS
# ═══════════════════════════════════════════════════════════════════════════

DIRECTION_LONG_PAIR = "LONG_PAIR"   # Buy Y, Sell X
DIRECTION_SHORT_PAIR = "SHORT_PAIR"  # Sell Y, Buy X

# ═══════════════════════════════════════════════════════════════════════════
# TRADE STATUS
# ═══════════════════════════════════════════════════════════════════════════

STATUS_OPEN = "OPEN"
STATUS_CLOSED = "CLOSED"
STATUS_STOPPED = "STOPPED"

# ═══════════════════════════════════════════════════════════════════════════
# EXIT REASONS
# ═══════════════════════════════════════════════════════════════════════════

EXIT_TARGET = "TARGET"
EXIT_STOP_LOSS = "STOP_LOSS"
EXIT_END_OF_DAY = "END_OF_DAY"
EXIT_MANUAL = "MANUAL"

# ═══════════════════════════════════════════════════════════════════════════
# USER PROFILES
# ═══════════════════════════════════════════════════════════════════════════

PROFILE_CONSERVATIVE = "conservative"
PROFILE_MODERATE = "moderate"
PROFILE_AGGRESSIVE = "aggressive"
