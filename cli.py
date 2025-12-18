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
from reporting.daily_report import generate_daily_report

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
        profile, margins, _, positions = fetch_account_snapshot()
        if profile:
            print(f"👤 User: {profile.get('user_id')} | {profile.get('user_name')}")
            print(f"💰 Cash: ₹{margins.get('net', 0):,.2f}")
            print(f"📉 Open Positions: {len(positions.get('net', []))}")
    except Exception as e:
        print(f"❌ Error: {e}")

def cmd_download(args):
    """
    Flexible Downloader:
    1. --symbol "INFY" -> Single symbol
    2. --file "data/artifacts/symbols.txt" -> List from file
    """
    print(f"\n--- ⬇️ DOWNLOADING DATA ({args.interval}) ---")
    
    symbols = []
    if args.file and os.path.exists(args.file):
        print(f"📄 Reading symbols from file: {args.file}")
        with open(args.file, "r") as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        print("❌ Error: Provide --symbol <NAME> or --file <PATH>")
        return

    # Use the Data Manager from Infrastructure
    download_historical_data(symbols, args.from_date, args.to_date, args.interval)

# --- RESEARCH LAB COMMANDS ---
def cmd_scan_momentum(_args):
    from research_lab.scan_momentum import scan_momentum
    scan_momentum()

def cmd_backtest_momentum(_args):
    from research_lab.backtest_momentum import run_backtest
    run_backtest()

def cmd_scan_pairs(_args):
    from research_lab.scan_pairs import scan_pairs
    scan_pairs()

def cmd_backtest_pairs(_args):
    from research_lab.backtest_pairs import backtest_pairs
    backtest_pairs()

# --- TRADING FLOOR COMMANDS ---
def cmd_engine(args):
    from trading_floor.engine import TradingEngine
    engine = TradingEngine(mode=args.mode)
    engine.start()

def cmd_report(args):
    generate_daily_report()

def main():
    parser = argparse.ArgumentParser(description="Financial Agent CLI")
    subparsers = parser.add_subparsers(title="command", dest="command", required=True)

    # 1. AUTH & BROKER
    subparsers.add_parser("login").set_defaults(func=cmd_login)
    p_tok = subparsers.add_parser("token")
    p_tok.add_argument("--request_token", required=True)
    p_tok.set_defaults(func=cmd_token)
    subparsers.add_parser("account").set_defaults(func=cmd_account)

    # 2. DATA
    p_dl = subparsers.add_parser("download", help="Download Historical Data")
    p_dl.add_argument("--symbol", help="Single Symbol (e.g. RELIANCE)")
    p_dl.add_argument("--file", help="Path to symbols file")
    p_dl.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    p_dl.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    p_dl.add_argument("--interval", default="5m", help="5m, day")
    p_dl.set_defaults(func=cmd_download)

    # 3. RESEARCH LAB
    subparsers.add_parser("scan_momentum").set_defaults(func=cmd_scan_momentum)
    subparsers.add_parser("backtest_momentum").set_defaults(func=cmd_backtest_momentum)
    subparsers.add_parser("scan_pairs").set_defaults(func=cmd_scan_pairs)
    subparsers.add_parser("backtest_pairs").set_defaults(func=cmd_backtest_pairs)

    # 4. TRADING FLOOR
    p_eng = subparsers.add_parser("engine")
    p_eng.add_argument("--mode", default="paper", choices=["paper", "live"])
    p_eng.set_defaults(func=cmd_engine)

    # 5. REPORT Command
    parser_report = subparsers.add_parser('report', help='Generate Daily PnL Report')
    parser_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
