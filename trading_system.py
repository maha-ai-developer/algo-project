#!/usr/bin/env python3
"""
Kite-connected MA + MACD + RSI trading system (terminal only).

Features
--------
- Backtest on CSV (daily / intraday / weekly) using MA + MACD + RSI strategy.
- Connects to Zerodha Kite via official `kiteconnect` client.
- Shows profile, funds / margins, holdings, positions.
- (Optional, requires DATA subscription) Download historical candlesticks.
- Position sizing: allocate % of your account to a trade.
- Optional auto-order placement with explicit user control.

Commands
--------
1) Login flow (once per day/session):
   python trading_system.py login-url
   python trading_system.py generate-token --request-token XYZ

2) See account + funds:
   python trading_system.py account

3) Backtest on CSV:
   python trading_system.py backtest --input data/SBIN_daily.csv

4) Download candles (needs full data plan, NOT Personal):
   python trading_system.py historical --instrument-token 779521 \
       --from-date 2024-01-01 --to-date 2025-01-01 \
       --interval day --output data/SBIN_daily.csv

5) Use CSV + funds to size position and (optionally) place order:
   python trading_system.py live \
       --input data/SBIN_daily.csv \
       --symbol SBIN --exchange NSE \
       --product CNC --capital-pct 10 \
       --place-order   # without this flag, it will NOT trade
"""

import argparse
import json
import os
import sys
from datetime import datetime

from typing import List, Optional

import numpy as np
import pandas as pd

from kiteconnect import KiteConnect  # pip install kiteconnect


# ----------------- CONFIG & KITE SETUP -----------------


CONFIG_PATH = "config.json"


def load_config(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise SystemExit(
            f"[ERROR] Config file not found: {path}\n"
            "Create it from config.example.json and fill api_key + api_secret."
        )
    with open(path, "r") as f:
        cfg = json.load(f)
    if not cfg.get("api_key") or not cfg.get("api_secret"):
        raise SystemExit("[ERROR] config.json must contain api_key and api_secret.")
    return cfg


def save_config(cfg: dict, path: str = CONFIG_PATH) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Saved updated config to {path}")


def init_kite(cfg: dict) -> KiteConnect:
    kite = KiteConnect(api_key=cfg["api_key"])
    access_token = cfg.get("access_token")
    if access_token:
        kite.set_access_token(access_token)
    else:
        print(
            "[WARN] No access_token in config. "
            "Run `login-url` and then `generate-token` first."
        )
    return kite


# ----------------- INDICATOR HELPERS -----------------


def normalize_col(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_to_original = {normalize_col(c): c for c in df.columns}
    for norm, original in norm_to_original.items():
        for cand in candidates:
            if cand == norm or cand in norm:
                return original
    return None


def detect_structure(df: pd.DataFrame):
    """Detect date and close columns for flexible CSV formats."""
    date_col = find_column(df, ["date", "timestamp", "trading_date", "time"])
    if date_col is None:
        raise ValueError("Could not find a Date/Timestamp column in the CSV.")

    close_col = find_column(
        df,
        [
            "close",
            "close_price",
            "closing_price",
            "closeprice",
            "last_price",
            "last_traded_price",
            "ltp",
        ],
    )
    if close_col is None:
        raise ValueError("Could not find a Close/Last Price column in the CSV.")

    volume_col = find_column(
        df,
        [
            "total_traded_quantity",
            "volume",
            "total_trades",
            "no_of_shrs",
            "traded_quantity",
        ],
    )
    # volume_col can be None
    return date_col, close_col, volume_col


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(close: pd.Series):
    """Classic MACD(12,26,9)."""
    ema_fast = compute_ema(close, 12)
    ema_slow = compute_ema(close, 26)
    macd = ema_fast - ema_slow
    signal = compute_ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Classic Wilder-style RSI (14). See RSI docs. :contentReference[oaicite:1]{index=1}
    """
    delta = close.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_series = pd.Series(gain, index=close.index)
    loss_series = pd.Series(loss, index=close.index)

    avg_gain = gain_series.rolling(window=period, min_periods=period).mean()
    avg_loss = loss_series.rolling(window=period, min_periods=period).mean()

    for i in range(period, len(close)):
        avg_gain.iat[i] = (avg_gain.iat[i - 1] * (period - 1) + gain_series.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i - 1] * (period - 1) + loss_series.iat[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_row_signal(row) -> str:
    """
    Generate Buy/Sell/Hold from last row using:
    - SMA20 / SMA50 trend
    - MACD vs signal
    - RSI band
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
    """Preprocess + compute indicators + signal on a price DataFrame."""
    date_col, close_col, volume_col = detect_structure(df)

    df = df.copy()
    df.rename(columns={date_col: "date", close_col: "close"}, inplace=True)
    if volume_col is not None:
        df.rename(columns={volume_col: "volume"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
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

    # Volume filter (if available)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["vol_sma20"] = df["volume"].rolling(window=20, min_periods=20).mean()
        df["high_volume"] = df["volume"] > (0.8 * df["vol_sma20"])
    else:
        df["high_volume"] = True

    df["signal"] = df.apply(generate_row_signal, axis=1)
    return df


# ----------------- POSITION SIZING -----------------


def compute_quantity(net_equity: float, price: float, capital_pct: float) -> int:
    """
    Simple position sizing: allocate % of net equity to trade.

    net_equity   : total net balance (e.g., from /user/margins equity.net) :contentReference[oaicite:2]{index=2}
    price        : last close / entry price
    capital_pct  : percentage of net_equity to use in this one trade
    """
    if price <= 0 or net_equity <= 0 or capital_pct <= 0:
        return 0
    capital = net_equity * (capital_pct / 100.0)
    qty = int(capital // price)
    return max(qty, 0)


# ----------------- COMMAND IMPLEMENTATIONS -----------------


def cmd_login_url(args):
    cfg = load_config()
    kite = KiteConnect(api_key=cfg["api_key"])
    url = kite.login_url()
    print("\n[LOGIN URL]")
    print(url)
    print(
        "\n1. Open this URL in your browser."
        "\n2. Login with your Zerodha account."
        "\n3. After login, you will be redirected to your Redirect URL with ?request_token=XXXX."
        "\n4. Copy that request_token and run:\n"
        "   python trading_system.py generate-token --request-token YOUR_TOKEN\n"
    )


def cmd_generate_token(args):
    cfg = load_config()
    kite = KiteConnect(api_key=cfg["api_key"])
    print("[INFO] Exchanging request_token for access_token...")
    data = kite.generate_session(args.request_token, api_secret=cfg["api_secret"])
    access_token = data["access_token"]
    cfg["access_token"] = access_token
    # store some optional info too
    cfg["user_id"] = data.get("user_id")
    cfg["public_token"] = data.get("public_token")
    save_config(cfg)
    print(f"[SUCCESS] Got access_token for user {data.get('user_id')}.")
    print("You can now run: python trading_system.py account")


def cmd_account(args):
    cfg = load_config()
    kite = init_kite(cfg)

    print("[INFO] Fetching profile...")
    profile = kite.profile()  # /user/profile :contentReference[oaicite:3]{index=3}

    print("[INFO] Fetching margins...")
    margins = kite.margins()  # /user/margins (equity + commodity) :contentReference[oaicite:4]{index=4}

    print("[INFO] Fetching holdings & positions...")
    holdings = kite.holdings()   # /portfolio/holdings :contentReference[oaicite:5]{index=5}
    positions = kite.positions()  # /portfolio/positions :contentReference[oaicite:6]{index=6}

    print("\n================ ACCOUNT SUMMARY ================")
    print(f"User ID   : {profile.get('user_id')}")
    print(f"Name      : {profile.get('user_name')}")
    print(f"Email     : {profile.get('email')}")
    print(f"Broker    : {profile.get('broker')}")
    print(f"Exchanges : {', '.join(profile.get('exchanges', []))}")
    print(f"Products  : {', '.join(profile.get('products', []))}")

    print("\n---------------- FUNDS (MARGINS) ----------------")
    for seg in ["equity", "commodity"]:
        if seg in margins:
            seg_data = margins[seg]
            net = seg_data.get("net")
            avail = seg_data.get("available", {})
            util = seg_data.get("utilised", {})
            print(f"\nSegment   : {seg}")
            print(f"  Enabled : {seg_data.get('enabled')}")
            print(f"  Net     : {net}")
            print(f"  Cash    : {avail.get('cash')}")
            print(f"  Opening : {avail.get('opening_balance')}")
            print(f"  Debits  : {util.get('debits')}")
            print(f"  Span    : {util.get('span')}, Exposure: {util.get('exposure')}")

    print("\n---------------- HOLDINGS ----------------")
    print(f"Total holdings: {len(holdings)}")
    if holdings:
        print("Top 5 holdings:")
        for h in holdings[:5]:
            print(
                f"  {h['tradingsymbol']} @ {h['average_price']} qty={h['quantity']} "
                f"ltp={h['last_price']} pnl={h['pnl']}"
            )

    print("\n---------------- POSITIONS ----------------")
    day = positions.get("day", [])
    net = positions.get("net", [])
    print(f"Intraday positions count: {len(day)}")
    print(f"Net positions count     : {len(net)}")

    if net:
        print("Top 5 net positions:")
        for p in net[:5]:
            print(
                f"  {p['tradingsymbol']} {p['quantity']} @ {p['average_price']} "
                f"pnl={p['pnl']}"
            )

    print("=================================================\n")


def load_and_analyse_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] Input file not found: {path}")
    df = pd.read_csv(path)
    df = apply_indicators(df)
    df_clean = df.dropna(subset=["sma20", "sma50", "macd", "macd_signal", "rsi"])
    if df_clean.empty:
        raise SystemExit("[ERROR] Not enough data to compute indicators (need >50 bars).")
    return df_clean


def print_latest_snapshot(df_clean: pd.DataFrame, title: str = ""):
    latest = df_clean.iloc[-1]
    if title:
        print(title)
    cols_to_show = [
        "date",
        "close",
        "sma20",
        "sma50",
        "macd",
        "macd_signal",
        "rsi",
        "high_volume",
        "signal",
    ]
    print("\nClean snapshot (last 5 rows):")
    print(df_clean[cols_to_show].tail(5).to_string(index=False))
    print("\nLatest row indicators:")
    print(f"  Date         : {latest['date']}")
    print(f"  Close        : {latest['close']:.2f}")
    print(f"  SMA20 / SMA50: {latest['sma20']:.2f} / {latest['sma50']:.2f}")
    print(f"  MACD / Signal: {latest['macd']:.4f} / {latest['macd_signal']:.4f}")
    print(f"  RSI(14)      : {latest['rsi']:.2f}")
    print(f"  High volume? : {bool(latest['high_volume'])}")
    print(f"  Strategy sig : {latest['signal']}")
    print()
    return latest


def cmd_backtest(args):
    df_clean = load_and_analyse_csv(args.input)
    latest = print_latest_snapshot(df_clean, title="===== BACKTEST (OFFLINE CSV) =====")
    print(f"FINAL BACKTEST SIGNAL on last bar: >>> {latest['signal']} <<<\n")


def cmd_historical(args):
    """
    Download historical candles from Kite (requires MARKET DATA subscription, not just Personal).
    This will likely fail with 403 TokenException on Personal if data APIs are disabled. :contentReference[oaicite:7]{index=7}
    """
    cfg = load_config()
    kite = init_kite(cfg)

    from_date = datetime.strptime(args.from_date, "%Y-%m-%d")
    to_date = datetime.strptime(args.to_date, "%Y-%m-%d")

    print("[INFO] Requesting historical candles...")
    data = kite.historical_data(
        instrument_token=args.instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval=args.interval,
        continuous=False,
        oi=False,
    )
    if not data:
        raise SystemExit("[ERROR] historical_data returned empty list.")

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[SUCCESS] Saved {len(df)} candles to {args.output}")


def cmd_live(args):
    """
    LIVE-ish mode:
    - Uses last bar from CSV (daily / intraday / weekly) as signal source.
    - Reads your funds from Kite.
    - Calculates quantity using capital_pct.
    - Shows a proposed order.
    - Only places order if --place-order flag is set.
    """
    cfg = load_config()
    kite = init_kite(cfg)

    # 1) Analyse CSV
    df_clean = load_and_analyse_csv(args.input)
    latest = print_latest_snapshot(df_clean, title="===== LIVE ANALYSIS (CSV + ACCOUNT) =====")
    signal = latest["signal"]
    close_price = float(latest["close"])

    # 2) Funds from Kite
    margins = kite.margins()
    equity_seg = margins.get("equity", {})
    net_equity = float(equity_seg.get("net", 0.0))

    print("---- FUNDS ----")
    print(f"Net equity funds: {net_equity}")
    print(f"Risk capital %  : {args.capital_pct}%")

    # 3) Compute quantity based on % of funds
    qty = compute_quantity(net_equity, close_price, args.capital_pct)

    if qty <= 0:
        print(
            "[WARN] Computed quantity <= 0. Either funds too low or capital_pct too small "
            "for this price."
        )
        return

    # 4) Decide direction from signal
    if signal == "BUY":
        transaction_type = "BUY"
    elif signal == "SELL":
        transaction_type = "SELL"
    else:
        print("[INFO] Strategy signal is HOLD. No trade suggested.")
        return

    est_capital = qty * close_price

    print("\n---- PROPOSED ORDER ----")
    print(f"Symbol         : {args.symbol}")
    print(f"Exchange       : {args.exchange}")
    print(f"Product        : {args.product}")
    print(f"Order type     : {args.order_type}")
    print(f"Transaction    : {transaction_type}")
    print(f"Qty            : {qty}")
    print(f"Entry (approx) : {close_price}")
    print(f"Capital used   : ~{est_capital:.2f} ({est_capital / net_equity * 100:.2f}% of equity)")
    print("\nNOTE: This uses last CSV close as proxy for entry price.")

    if not args.place_order:
        print(
            "\n[SAFE MODE] --place-order NOT given. No order placed.\n"
            "If you are happy with this plan, run the same command with --place-order to execute."
        )
        return

    # 5) Confirm interactively as an extra safety layer
    confirm = input("\nType 'YES' to place this order via Kite: ").strip()
    if confirm != "YES":
        print("[INFO] Confirmation failed. Not placing order.")
        return

    # 6) Place order via Kite
    print("\n[INFO] Placing order via Kite...")
    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=args.exchange,
            tradingsymbol=args.symbol,
            transaction_type=transaction_type,
            quantity=qty,
            product=args.product,
            order_type=args.order_type,
            price=None if args.order_type == "MARKET" else close_price,
            validity=kite.VALIDITY_DAY,
        )
        print(f"[SUCCESS] Order placed. ID: {order_id}")
    except Exception as e:
        print(f"[ERROR] Order placement failed: {e}")


# ----------------- MAIN / CLI -----------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kite-connected MA + MACD + RSI trading system (terminal).",
    )
    subparsers = parser.add_subparsers(dest="command")

    # login-url
    p_login = subparsers.add_parser("login-url", help="Show Kite login URL for request_token.")
    p_login.set_defaults(func=cmd_login_url)

    # generate-token
    p_token = subparsers.add_parser(
        "generate-token",
        help="Exchange request_token + api_secret for access_token and save to config.json.",
    )
    p_token.add_argument("--request-token", required=True, help="request_token from login redirect.")
    p_token.set_defaults(func=cmd_generate_token)

    # account
    p_account = subparsers.add_parser("account", help="Show profile, margins, holdings, positions.")
    p_account.set_defaults(func=cmd_account)

    # backtest
    p_backtest = subparsers.add_parser("backtest", help="Backtest strategy on CSV (offline).")
    p_backtest.add_argument("--input", "-i", required=True, help="Path to OHLC CSV file.")
    p_backtest.set_defaults(func=cmd_backtest)

    # historical (requires data plan, NOT personal)
    p_hist = subparsers.add_parser(
        "historical",
        help="Download historical candles via Kite (needs data subscription).",
    )
    p_hist.add_argument(
        "--instrument-token",
        type=int,
        required=True,
        help="Instrument token (e.g., SBIN EQ is often 779521).",
    )
    p_hist.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    p_hist.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    p_hist.add_argument(
        "--interval",
        default="day",
        help="Kite interval (e.g., minute, 5minute, 15minute, 60minute, day, week).",
    )
    p_hist.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV path for saving candles.",
    )
    p_hist.set_defaults(func=cmd_historical)

    # live
    p_live = subparsers.add_parser(
        "live",
        help="Use CSV + account funds to size trade and (optionally) place order.",
    )
    p_live.add_argument("--input", "-i", required=True, help="Path to CSV used for latest signal.")
    p_live.add_argument("--symbol", required=True, help="Trading symbol, e.g., SBIN.")
    p_live.add_argument(
        "--exchange",
        default="NSE",
        help="Exchange (NSE, BSE, NFO, etc.).",
    )
    p_live.add_argument(
        "--product",
        default="CNC",
        help="Product (CNC, MIS, NRML, etc.).",
    )
    p_live.add_argument(
        "--order-type",
        default="MARKET",
        help="Order type (MARKET, LIMIT).",
    )
    p_live.add_argument(
        "--capital-pct",
        type=float,
        default=10.0,
        help="Percentage of equity funds to allocate to this single trade.",
    )
    p_live.add_argument(
        "--place-order",
        action="store_true",
        help="Actually send order to Kite. Without this flag, just show the plan.",
    )
    p_live.set_defaults(func=cmd_live)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
