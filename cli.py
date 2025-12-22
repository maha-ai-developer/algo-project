#!/usr/bin/env python
import argparse
import sys
import os

# Add root to path so we can import 'infrastructure', 'strategies', etc.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import infrastructure.config as config
from infrastructure.broker.kite_auth import generate_login_url, exchange_request_token
from infrastructure.broker.kite_positions import fetch_account_snapshot
from infrastructure.data.data_manager import download_historical_data

# ===========================================================
# COMMAND HANDLERS
# ===========================================================

def cmd_login(_args):
    try:
        url = generate_login_url()
        print("\n--- 🔐 ZERODHA LOGIN ---")
        print(f"1. Open: {url}")
        print("2. Login & Copy 'request_token' from URL")
        print("3. Run: python cli.py token --request_token <TOKEN>\n")
    except Exception as e:
        print(f"❌ Error: {e}")

def cmd_token(args):
    print("\n--- 🔄 EXCHANGING TOKEN ---")
    try:
        token = exchange_request_token(args.request_token)
        if token:
            print(f"✅ Access Token saved to: {config.CONFIG_FILE}")
        else:
            print("❌ Failed.")
    except Exception as e:
        print(f"❌ Error: {e}")

def cmd_account(_args):
    print("\n--- 🏦 ACCOUNT STATUS ---")
    try:
        profile, margins, _, _ = fetch_account_snapshot()
        print(f"👤 Name: {profile.get('user_name')}")
        print(f"💰 Cash: ₹{margins.get('net', 0):,.2f}")
    except Exception as e:
        print(f"❌ Error: {e}")

def cmd_download(args):
    if args.file:
        with open(args.file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        print("❌ Specify --symbol or --file")
        return

    print(f"⬇️ Downloading {len(symbols)} symbols...")
    download_historical_data(symbols, args.from_date, args.to_date, args.interval)

# --- RESEARCH COMMANDS (UPDATED) ---

def cmd_scan_fundamental(_args):
    from research_lab.scan_fundamental import run_fundamental_scan
    run_fundamental_scan()

def cmd_sector_analysis(_args):
    from research_lab.sector_analysis import run_sector_analysis
    run_sector_analysis()

def cmd_scan_pairs(_args):
    from research_lab.scan_pairs import scan_pairs
    scan_pairs()

def cmd_backtest_pairs(_args):
    # POINTING TO THE NEW GUARDIAN BACKTEST
    from research_lab.backtest_pairs import run_pro_backtest
    run_pro_backtest()

# --- TRADING FLOOR COMMANDS ---

def cmd_engine(args):
    from trading_floor.engine import TradingEngine
    # The Engine now handles everything (Guardian, Risk, Execution)
    engine = TradingEngine(mode=args.mode)
    engine.run()

# ===========================================================
# MAIN PARSER
# ===========================================================

def main():
    parser = argparse.ArgumentParser(description="Algo Trading CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. AUTH
    subparsers.add_parser("login", help="Generate Login URL").set_defaults(func=cmd_login)
    p_tok = subparsers.add_parser("token", help="Exchange Request Token")
    p_tok.add_argument("--request_token", required=True)
    p_tok.set_defaults(func=cmd_token)
    subparsers.add_parser("account", help="Show Margins").set_defaults(func=cmd_account)

    # 2. DATA
    p_dl = subparsers.add_parser("download", help="Download Historical Data")
    p_dl.add_argument("--symbol", help="Single Symbol (e.g. RELIANCE)")
    p_dl.add_argument("--file", help="Path to symbols file")
    p_dl.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    p_dl.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    p_dl.add_argument("--interval", default="5m", help="5m, day")
    p_dl.set_defaults(func=cmd_download)

    # 3. RESEARCH LAB
    subparsers.add_parser("scan_fundamental", help="1. Financial Health Check").set_defaults(func=cmd_scan_fundamental)
    subparsers.add_parser("sector_analysis", help="2. Sector Deep Dive").set_defaults(func=cmd_sector_analysis)
    subparsers.add_parser("scan_pairs", help="3. Find Cointegrated Pairs (Method 2)").set_defaults(func=cmd_scan_pairs)
    subparsers.add_parser("backtest_pairs", help="4. Run Pro Backtest with Guardian").set_defaults(func=cmd_backtest_pairs)
    
    # 4. TRADING FLOOR
    p_eng = subparsers.add_parser("engine")
    p_eng.add_argument("--mode", choices=["PAPER", "LIVE"], default="PAPER")
    p_eng.set_defaults(func=cmd_engine)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
