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
    from trading_floor.engine import create_engine
    # Use factory function for optimized engine (v2.0)
    engine = create_engine(mode=args.mode, use_websocket=args.websocket)
    engine.run()

# --- FUTURES UTILITIES ---

def cmd_futures_info(args):
    """Get futures contract information for a symbol."""
    from infrastructure.data.futures_utils import get_contract_info, get_all_expiries
    from tabulate import tabulate
    
    print(f"\n--- 📊 FUTURES INFO: {args.symbol} ---")
    
    # Get contract info (uses cached instruments or fallback)
    info = get_contract_info(args.symbol.upper(), price=0)
    
    print(f"\n📦 Contract Details:")
    print(f"   Spot Symbol:    {info['spot_symbol']}")
    print(f"   Futures Symbol: {info['futures_symbol']}")
    print(f"   Lot Size:       {info['lot_size']:,}")
    print(f"   Expiry:         {info['expiry']}")
    print(f"   Margin:         {info['margin_pct']}")
    print(f"   Data Source:    {info['data_source']}")
    
    # Get all expiries
    expiries = get_all_expiries(args.symbol.upper())
    if expiries:
        print(f"\n📅 Available Expiries:")
        for e in expiries:
            print(f"   {e['month']:4} | {e['symbol']} | Lot: {e['lot_size']}")
    
    print(f"\n💡 Tip: Run 'python cli.py refresh_instruments' to update from Kite API")

def cmd_download_futures(args):
    """Download futures historical data."""
    from infrastructure.data.futures_utils import get_futures_details, download_futures_historical
    from infrastructure.broker.kite_auth import get_kite
    import pandas as pd
    
    print(f"\n--- ⬇️ DOWNLOADING FUTURES DATA: {args.symbol} ---")
    
    try:
        kite = get_kite()
    except Exception as e:
        print(f"❌ Failed to connect to Kite: {e}")
        print("   Run 'python cli.py login' first")
        return
    
    # Get futures details
    details = get_futures_details(args.symbol.upper(), kite)
    if not details:
        print(f"❌ No futures found for {args.symbol}")
        return
    
    print(f"   Symbol: {details['symbol']}")
    print(f"   Lot Size: {details['lot_size']}")
    print(f"   Expiry: {details['expiry']}")
    print(f"   Continuous: {args.continuous}")
    
    # Download data
    data = download_futures_historical(
        args.symbol.upper(),
        args.from_date,
        args.to_date,
        args.interval,
        continuous=args.continuous,
        kite=kite
    )
    
    if data:
        df = pd.DataFrame(data)
        filename = f"{details['symbol']}_{args.interval}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✅ Saved {len(df)} candles to {filepath}")
    else:
        print("❌ Failed to download data")

def cmd_refresh_instruments(_args):
    """Refresh NFO instrument cache from Kite API."""
    from infrastructure.data.futures_utils import refresh_instrument_cache
    from infrastructure.broker.kite_auth import get_kite
    
    print("\n--- 🔄 REFRESHING INSTRUMENT CACHE ---")
    
    try:
        kite = get_kite()
        count = refresh_instrument_cache(kite)
        print(f"\n✅ Loaded {count:,} NFO instruments to cache")
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("   Run 'python cli.py login' first")

def cmd_download_all_futures(args):
    """Download futures data for all candidate pairs (36 pairs from pairs_candidates.json)."""
    from infrastructure.data.futures_utils import get_futures_details, download_futures_historical
    from infrastructure.broker.kite_auth import get_kite
    import pandas as pd
    import json
    import os
    from datetime import datetime, timedelta
    
    print("\n--- ⬇️ DOWNLOADING FUTURES DATA FOR ALL CANDIDATE PAIRS ---")
    
    # Load ALL candidates (not just winners)
    if not os.path.exists(config.PAIRS_CANDIDATES_FILE):
        print(f"❌ No candidates found at {config.PAIRS_CANDIDATES_FILE}")
        print("   Run 'python cli.py scan_pairs' first.")
        return
    
    with open(config.PAIRS_CANDIDATES_FILE, 'r') as f:
        pairs = json.load(f)
    
    # Get unique symbols
    symbols = set()
    for pair in pairs:
        symbols.add(pair['leg1'])
        symbols.add(pair['leg2'])
    
    symbols = sorted(symbols)
    print(f"   📊 Found {len(pairs)} pairs with {len(symbols)} unique symbols")
    
    # Connect to Kite
    try:
        kite = get_kite()
    except Exception as e:
        print(f"❌ Failed to connect to Kite: {e}")
        print("   Run 'python cli.py login' first")
        return
    
    # Dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    print(f"   📅 Date range: {start_date} to {end_date}")
    print(f"   📁 Saving to: {config.DATA_DIR}")
    print()
    
    success = 0
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"   [{i}/{len(symbols)}] {symbol}...", end=" ")
        
        try:
            details = get_futures_details(symbol, kite)
            if not details:
                print("❌ No futures found")
                failed.append(symbol)
                continue
            
            data = download_futures_historical(
                symbol,
                start_date,
                end_date,
                "day",
                continuous=True,
                kite=kite
            )
            
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                filename = f"{details['symbol']}_day.csv"
                filepath = os.path.join(config.DATA_DIR, filename)
                df.to_csv(filepath, index=False)
                print(f"✅ {len(df)} candles → {filename}")
                success += 1
            else:
                print("❌ No data returned")
                failed.append(symbol)
                
        except Exception as e:
            print(f"❌ {e}")
            failed.append(symbol)
    
    print(f"\n{'='*50}")
    print(f"   ✅ Downloaded: {success}/{len(symbols)} symbols")
    if failed:
        print(f"   ❌ Failed: {', '.join(failed)}")

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
    
    # 5. FUTURES UTILITIES
    p_fut = subparsers.add_parser("futures_info", help="Get futures contract info for a symbol")
    p_fut.add_argument("--symbol", required=True, help="Symbol (e.g. SBIN)")
    p_fut.set_defaults(func=cmd_futures_info)
    
    p_futdl = subparsers.add_parser("download_futures", help="Download futures data")
    p_futdl.add_argument("--symbol", required=True, help="Spot symbol (e.g. SBIN)")
    p_futdl.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    p_futdl.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    p_futdl.add_argument("--interval", default="day", help="day, minute")
    p_futdl.add_argument("--continuous", action="store_true", help="Use continuous data (handles rollover)")
    p_futdl.set_defaults(func=cmd_download_futures)
    
    subparsers.add_parser("refresh_instruments", help="Refresh NFO instrument cache").set_defaults(func=cmd_refresh_instruments)
    subparsers.add_parser("download_all_futures", help="Download futures data for all winning pairs").set_defaults(func=cmd_download_all_futures)
    
    # 4. TRADING FLOOR (Updated for v2.0)
    p_eng = subparsers.add_parser("engine", help="Run Trading Engine v2.0")
    p_eng.add_argument("--mode", choices=["PAPER", "LIVE"], default="PAPER")
    p_eng.add_argument("--websocket", action="store_true", help="Enable real-time WebSocket updates")
    p_eng.set_defaults(func=cmd_engine)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

