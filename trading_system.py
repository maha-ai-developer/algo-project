#!/usr/bin/env python3
"""
Simple terminal-based trading system using Moving Averages, MACD and RSI.

Usage:
    python trading_system.py --input Quote-Equity-SBIN-EQ-03-12-2024-03-12-2025.csv
    python trading_system.py --input data.csv --plot
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------- Utility functions ----------

def normalize_col(name: str) -> str:
    return (
        name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
    )


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Try to find a column whose normalized name matches any of the candidate keys.
    """
    norm_to_original = {normalize_col(c): c for c in df.columns}
    for norm, original in norm_to_original.items():
        for cand in candidates:
            if cand == norm or cand in norm:
                return original
    return None


def detect_structure(df: pd.DataFrame):
    """
    Detect date, close and volume columns for common Indian / global equity CSVs.
    """
    date_col = find_column(df, ["date", "timestamp", "trading_date"])
    if date_col is None:
        raise ValueError("Could not find a Date column in the CSV.")

    close_col = find_column(
        df,
        [
            "close", "close_price", "closing_price",
            "closeprice", "last_price", "last_traded_price",
        ],
    )
    if close_col is None:
        raise ValueError("Could not find a Close/Last Price column in the CSV.")

    volume_col = find_column(
        df,
        [
            "total_traded_quantity", "volume", "total_trades",
            "no_of_shrs", "traded_quantity",
        ],
    )
    # volume_col can be None; we treat it as optional

    return date_col, close_col, volume_col


# ---------- Indicator calculations ----------

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(close: pd.Series):
    """
    Standard MACD(12,26,9).
    """
    ema_fast = compute_ema(close, 12)
    ema_slow = compute_ema(close, 26)
    macd = ema_fast - ema_slow
    signal = compute_ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Classic Wilder-style RSI. :contentReference[oaicite:4]{index=4}
    """
    delta = close.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_series = pd.Series(gain, index=close.index)
    loss_series = pd.Series(loss, index=close.index)

    avg_gain = gain_series.rolling(window=period, min_periods=period).mean()
    avg_loss = loss_series.rolling(window=period, min_periods=period).mean()

    # Wilder smoothing for subsequent values
    for i in range(period, len(close)):
        avg_gain.iat[i] = (avg_gain.iat[i - 1] * (period - 1) + gain_series.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i - 1] * (period - 1) + loss_series.iat[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------- Signal engine ----------

def generate_row_signal(row) -> str:
    """
    Generate Buy/Sell/Hold for a single row (latest bar).
    """
    close = row["close"]
    sma20 = row["sma20"]
    sma50 = row["sma50"]
    macd = row["macd"]
    macd_sig = row["macd_signal"]
    rsi = row["rsi"]

    if np.isnan([sma20, sma50, macd, macd_sig, rsi]).any():
        return "HOLD"

    uptrend = (close > sma50) and (sma20 > sma50)
    downtrend = (close < sma50) and (sma20 < sma50)

    bullish_mom = macd > macd_sig and 50 < rsi < 70
    bearish_mom = macd < macd_sig and 30 < rsi < 50

    if uptrend and bullish_mom:
        return "BUY"
    elif downtrend and bearish_mom:
        return "SELL"
    else:
        return "HOLD"


def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    date_col, close_col, volume_col = detect_structure(df)

    # Rename for internal consistency
    df = df.copy()
    df.rename(columns={date_col: "date", close_col: "close"}, inplace=True)
    if volume_col is not None:
        df.rename(columns={volume_col: "volume"}, inplace=True)

    # Parse dates and sort
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    # Ensure numeric close
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Drop rows with no close
    df = df.dropna(subset=["close"])

    # Moving averages
    df["sma20"] = compute_sma(df["close"], 20)
    df["sma50"] = compute_sma(df["close"], 50)
    df["ema12"] = compute_ema(df["close"], 12)
    df["ema26"] = compute_ema(df["close"], 26)

    # MACD
    macd, macd_signal, macd_hist = compute_macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # RSI
    df["rsi"] = compute_rsi(df["close"], 14)

    # Optional volume filter: require volume above 20-day average
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["vol_sma20"] = df["volume"].rolling(window=20, min_periods=20).mean()
        df["high_volume"] = df["volume"] > (0.8 * df["vol_sma20"])
    else:
        df["high_volume"] = True  # neutral if no volume data

    # Row-wise signal
    df["signal"] = df.apply(generate_row_signal, axis=1)

    return df


# ---------- Plotting ----------

def make_plot(df: pd.DataFrame, out_path: str):
    if not HAS_MPL:
        print("matplotlib is not installed; skipping plot.")
        return

    latest = df.iloc[-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"Price with SMA20 & SMA50 – up to {latest['date'].date()}")

    plt.plot(df["date"], df["close"], label="Close")
    plt.plot(df["date"], df["sma20"], label="SMA20")
    plt.plot(df["date"], df["sma50"], label="SMA50")

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved plot to {out_path}")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Terminal-based MA + MACD + RSI trading system."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to CSV file with historical OHLC data.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a PNG chart with price and moving averages.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    df = apply_indicators(df)

    # Drop rows where indicators are NaN (warm-up period)
    df_clean = df.dropna(subset=["sma20", "sma50", "macd", "macd_signal", "rsi"])

    if df_clean.empty:
        raise SystemExit("Not enough data to compute indicators. Add more rows.")

    latest = df_clean.iloc[-1]
    recommendation = latest["signal"]

    print("=" * 60)
    print(f"Data file     : {args.input}")
    print(f"Last date     : {latest['date'].date()}")
    print("-" * 60)
    print("Clean preprocessed snapshot (last 5 rows):")
    cols_to_show = [
        "date", "close", "sma20", "sma50", "macd", "macd_signal", "rsi", "high_volume",
        "signal",
    ]
    print(df_clean[cols_to_show].tail(5).to_string(index=False))
    print("-" * 60)
    print("Latest indicators:")
    print(f"  Close        : {latest['close']:.2f}")
    print(f"  SMA20 / SMA50: {latest['sma20']:.2f} / {latest['sma50']:.2f}")
    print(f"  MACD / Signal: {latest['macd']:.4f} / {latest['macd_signal']:.4f}")
    print(f"  RSI(14)      : {latest['rsi']:.2f}")
    print(f"  High volume? : {bool(latest['high_volume'])}")
    print("-" * 60)
    print(f"FINAL RECOMMENDATION: >>>> {recommendation} <<<<")
    print("=" * 60)

    if args.plot:
        out_path = os.path.splitext(args.input)[0] + "_chart.png"
        make_plot(df_clean, out_path)


if __name__ == "__main__":
    main()
