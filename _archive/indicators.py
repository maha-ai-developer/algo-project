# core/indicators.py
import pandas as pd
import numpy as np


# -------------------------------
# Basic MAs
# -------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# -------------------------------
# RSI
# -------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


# -------------------------------
# MACD
# -------------------------------
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -------------------------------
# SuperTrend (basic version)
# -------------------------------
def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Returns (supertrend_band, trend_direction)
    trend_direction is +1 for uptrend, -1 for downtrend.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    hl2 = (high + low) / 2.0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = 1
        else:
            if close.iloc[i] > supertrend.iloc[i - 1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < supertrend.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(lowerband.iloc[i], supertrend.iloc[i - 1])
            else:
                supertrend.iloc[i] = min(upperband.iloc[i], supertrend.iloc[i - 1])

    return supertrend, direction


# -------------------------------
# VWAP
# -------------------------------
def vwap(df: pd.DataFrame) -> pd.Series:
    """
    df must have: 'high','low','close','volume'
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


# -------------------------------
# Combined helper for backtest
# -------------------------------
def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a standard set of indicators to df for backtesting.

    Expects df with columns:
        'close', and ideally 'high','low','volume'
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must have 'close' column")

    c = df["close"]

    # Moving averages
    df["sma20"] = sma(c, 20)
    df["sma50"] = sma(c, 50)
    df["ema8"] = ema(c, 8)
    df["ema21"] = ema(c, 21)

    # MACD
    macd_line, signal_line, hist = macd(c)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    # RSI
    df["rsi14"] = rsi(c, 14)

    # Supertrend + VWAP (if we have high/low/volume)
    if {"high", "low", "volume"}.issubset(df.columns):
        st, st_dir = supertrend(df)
        df["supertrend"] = st
        df["supertrend_dir"] = st_dir
        df["vwap"] = vwap(df)
    else:
        df["supertrend"] = np.nan
        df["supertrend_dir"] = np.nan
        df["vwap"] = np.nan

    return df
