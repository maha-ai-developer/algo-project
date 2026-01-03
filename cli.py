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

def cmd_fetch_universe(_args):
    """Fetch futures universe from Kite NFO segment."""
    from infrastructure.data.fetch_futures_universe import fetch_futures_universe
    fetch_futures_universe()

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

def cmd_ai_advisor(_args):
    """Run AI Advisor for system-wide oversight."""
    from research_lab.ai_advisor import run_ai_advisor
    run_ai_advisor()

# --- TRADING FLOOR COMMANDS ---

def cmd_engine(args):
    from trading_floor.engine import create_engine
    # Use factory function for optimized engine (v2.0)
    engine = create_engine(mode=args.mode, use_websocket=args.websocket)
    engine.run()

def cmd_positions(args):
    """Live position tracker - Excel-style format with WebSocket streaming."""
    import time
    import json
    import threading
    from datetime import datetime
    from trading_floor.state import StateManager
    from infrastructure.broker.kite_auth import get_kite
    import infrastructure.config as config
    
    sm = StateManager()
    trades = sm.load()
    
    if not trades:
        print("\n📭 No active positions.\n")
        return
    
    try:
        kite = get_kite()
    except Exception as e:
        print(f"\n❌ Failed to connect to Kite: {e}")
        return
    
    # Load pairs config for sigma values
    pairs_config = {}
    if os.path.exists(config.PAIRS_CONFIG):
        with open(config.PAIRS_CONFIG, 'r') as f:
            for p in json.load(f):
                s_y = p.get('stock_y') or p.get('leg1')
                s_x = p.get('stock_x') or p.get('leg2')
                key = f"{s_y}-{s_x}"
                pairs_config[key] = p
    
    # Get symbols for LTP
    symbols = set()
    for pair_key in trades:
        s1, s2 = pair_key.split('-')
        symbols.add(s1)
        symbols.add(s2)
    
    # Get instrument tokens
    instruments = kite.instruments("NSE")
    inst_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
    token_to_symbol = {v: k for k, v in inst_map.items() if k in symbols}
    token_list = [inst_map[s] for s in symbols if s in inst_map]
    
    refresh_interval = args.interval if hasattr(args, 'interval') else 5
    use_websocket = args.websocket if hasattr(args, 'websocket') else False
    
    # Shared price dictionary (thread-safe updates)
    ltp_map = {}
    for s in symbols:
        ltp_map[s] = 0  # Initialize
    
    def display_positions():
        """Display the position tracker screen."""
        # Clear screen
        print("\033[H\033[J", end="")
        
        # Header
        now = datetime.now().strftime('%d %b %Y  %H:%M:%S')
        mode_str = "🔴 LIVE WebSocket" if use_websocket else "📡 REST Polling"
        print("="*72)
        print(f"{'📊 POSITION TRACKER':^72}")
        print(f"{'⏰ ' + now:^72}")
        print(f"{mode_str:^72}")
        print("="*72)
        
        grand_total = 0
        
        for pair_key, trade in trades.items():
            s1, s2 = pair_key.split('-')
            side = trade.get('side', 'N/A')
            beta = trade.get('beta', 1.0)
            intercept = trade.get('intercept', 0)
            entry_y = trade.get('entry_price_y', 0)
            entry_x = trade.get('entry_price_x', 0)
            qty_y = trade.get('q1', 0)
            qty_x = trade.get('q2', 0)
            entry_z = trade.get('entry_zscore', 0)
            entry_time = trade.get('entry_time', '')
            sector = trade.get('sector', 'FINANCE')
            
            # Get sigma from pairs_config
            pair_cfg = pairs_config.get(pair_key, {})
            sigma = pair_cfg.get('sigma', 50)
            
            # LIVE PRICES
            live_y = ltp_map.get(s1, 0) or entry_y
            live_x = ltp_map.get(s2, 0) or entry_x
            
            # Current Z-score
            residual = live_y - (beta * live_x + intercept)
            current_z = residual / sigma if sigma else 0
            
            # Beta Neutrality lot requirement
            lot_y = pair_cfg.get('lot_size_y', qty_y)
            lot_x = pair_cfg.get('lot_size_x', qty_x)
            for_1_lot = round(lot_y / beta) if beta else lot_y
            
            # ═══ PAIR DATA BOX ═══
            print()
            print("┌─ Pair Data ────────────────────────┬─ For Beta Neutrality ────────────┐")
            print(f"│ Dependent Stock (Y)     {s1:>10} │ Lot size of Y         {lot_y:>10,} │")
            print(f"│ Independent Stock (X)   {s2:>10} │ Lot size of X         {lot_x:>10,} │")
            print(f"│ Sector                  {sector:>10} │ For 1 lot of Y        {for_1_lot:>10,} │")
            print("└────────────────────────────────────┴──────────────────────────────────┘")
            
            # ═══ REGRESSION + SIGNAL BOX ═══
            entry_date = str(entry_time)[:10] if entry_time else "N/A"
            print()
            print("┌─ Regression Parameters ────────────┬─ Signal ─────────────────────────┐")
            print(f"│ Beta                    {beta:>10.4f} │ Entry Date            {entry_date:>10} │")
            print(f"│ Intercept               {intercept:>10.2f} │ Entry Price Y      ₹{entry_y:>10,.2f} │")
            print(f"│ Sigma                   {sigma:>10.4f} │ Entry Price X      ₹{entry_x:>10,.2f} │")
            print(f"│                                    │ Entry Z-Score       {entry_z:>10.4f} │")
            print("└────────────────────────────────────┴──────────────────────────────────┘")
            
            # ═══ TRADE EXECUTED vs CURRENT VALUES ═══
            print()
            print("┌─ Trade Executed ───────────────────┬─ Current Values (LIVE) ──────────┐")
            print(f"│ Fut (Y)              ₹{entry_y:>10,.2f} │ Fut (Y)            ₹{live_y:>10,.2f} │")
            print(f"│ Fut (X)              ₹{entry_x:>10,.2f} │ Fut (X)            ₹{live_x:>10,.2f} │")
            print(f"│ Z-Score               {entry_z:>10.4f} │ Z-Score             {current_z:>10.4f} │")
            print("└────────────────────────────────────┴──────────────────────────────────┘")
            
            # Calculate P&L
            if side == 'SHORT':
                pnl_y = (entry_y - live_y) * qty_y
                pnl_x = (live_x - entry_x) * qty_x
                pos_y, pos_x = 'Short', 'Long'
            else:
                pnl_y = (live_y - entry_y) * qty_y
                pnl_x = (entry_x - live_x) * qty_x
                pos_y, pos_x = 'Long', 'Short'
            
            total_pnl = pnl_y + pnl_x
            grand_total += total_pnl
            
            # ═══ P&L TABLE ═══
            print()
            print("┌─ P&L ───────────────────────────────────────────────────────────────┐")
            print("│ Stock              Position   Qty     Trade Price  Current   P&L   │")
            print("├────────────────────────────────────────────────────────────────────┤")
            print(f"│ {s1:<18} {pos_y:<8}  {qty_y:>5,}   ₹{entry_y:>9,.2f}  ₹{live_y:>8,.2f}  {pnl_y:>+8,.0f} │")
            print(f"│ {s2:<18} {pos_x:<8}  {qty_x:>5,}   ₹{entry_x:>9,.2f}  ₹{live_x:>8,.2f}  {pnl_x:>+8,.0f} │")
            print("├────────────────────────────────────────────────────────────────────┤")
            pnl_icon = "📈" if total_pnl >= 0 else "📉"
            print(f"│ {pnl_icon} TOTAL                                              {total_pnl:>+10,.0f} │")
            print("└────────────────────────────────────────────────────────────────────┘")
        
        # Grand Total
        if len(trades) > 1:
            print()
            print("="*72)
            g_icon = "📈" if grand_total >= 0 else "📉"
            print(f" {g_icon} GRAND TOTAL P&L: ₹{grand_total:+,.0f}".center(72))
            print("="*72)
        
        # Rules + Status
        print()
        print("┌─ Rules ───────────────────────────────────────────────────────────────┐")
        print("│ 1) Entry: Z-Score above +2.5 or below -2.5                            │")
        print("│ 2) Stop Loss: Z-Score hits ±3.0                                       │")
        print("│ 3) Target: Z-Score reaches ±1.0 (mean reversion)                      │")
        print("└────────────────────────────────────────────────────────────────────────┘")
        print(f"\n ⚡ Refresh: {refresh_interval}s  │  Ctrl+C to stop")
    
    # WebSocket mode
    if use_websocket:
        from kiteconnect import KiteTicker
        
        def on_ticks(ws, ticks):
            """Handle incoming WebSocket ticks."""
            for tick in ticks:
                token = tick['instrument_token']
                if token in token_to_symbol:
                    symbol = token_to_symbol[token]
                    ltp_map[symbol] = tick['last_price']
            display_positions()
        
        def on_connect(ws, response):
            """Subscribe to instruments on connect."""
            print(f"🔌 WebSocket connected! Subscribing to {len(token_list)} tokens...")
            ws.subscribe(token_list)
            ws.set_mode(ws.MODE_LTP, token_list)
        
        def on_close(ws, code, reason):
            print(f"🔴 WebSocket closed: {reason}")
        
        def on_error(ws, code, reason):
            print(f"⚠️ WebSocket error: {reason}")
        
        # Get API credentials
        with open(config.CONFIG_FILE, 'r') as f:
            creds = json.load(f).get('kite', {})
        
        kws = KiteTicker(creds['api_key'], creds['access_token'])
        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.on_error = on_error
        
        print("🔌 Starting WebSocket connection...")
        kws.connect(threaded=True)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            kws.close()
            print("\n\n✅ Position tracking stopped.")
    
    # REST polling mode (fallback)
    else:
        try:
            while True:
                # Get live quotes via REST
                try:
                    quotes = kite.quote([inst_map[s] for s in symbols if s in inst_map])
                    for s in symbols:
                        if s in inst_map:
                            token = inst_map[s]
                            if token in quotes:
                                ltp_map[s] = quotes[token]['last_price']
                except Exception as e:
                    print(f"⚠️ Quote error: {e}")
                
                display_positions()
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n✅ Position tracking stopped.")


# --- REPORTING COMMANDS ---

def cmd_pair_report(_args):
    """Generate pair data report with Z-scores and ADF values."""
    from reporting.pair_data_report import generate_pair_report
    generate_pair_report()

def cmd_analytics(args):
    """Generate trade analytics report."""
    from reporting.trade_analytics import generate_trade_analytics
    generate_trade_analytics(days_back=args.days)

def cmd_daily_report(_args):
    """Generate daily P&L report."""
    from reporting.daily_report import generate_daily_report
    generate_daily_report()

def cmd_ai_analysis(args):
    """Generate AI-powered post-trade analysis with Gemini."""
    from reporting.ai_analysis import generate_ai_analysis
    from reporting.ai_analysis import generate_ai_analysis
    generate_ai_analysis(days_back=args.days)

def cmd_analyze_backtest(_args):
    """Analyze backtest results with AI."""
    from reporting.ai_analysis import analyze_backtest_file
    analyze_backtest_file()

def cmd_news_scan(_args):
    """Scan active positions for corporate actions and critical news."""
    from trading_floor.news_monitor import scan_positions_for_news
    scan_positions_for_news()

def cmd_pair_stats(_args):
    """Display detailed regression statistics for all configured pairs."""
    import json
    import pandas as pd
    from strategies.stat_arb_bot import StatArbBot
    from infrastructure.broker.kite_auth import get_kite
    from infrastructure.data.cache import DataCache
    import infrastructure.config as config
    
    print(f"\n📊 --- PAIR REGRESSION STATISTICS ---")
    
    # Load pairs config
    if not os.path.exists(config.PAIRS_CONFIG):
        print(f"❌ No pairs config found: {config.PAIRS_CONFIG}")
        return
    
    with open(config.PAIRS_CONFIG, 'r') as f:
        pairs = json.load(f)
    
    print(f"   Loaded {len(pairs)} pairs from config\n")
    
    try:
        kite = get_kite()
        cache = DataCache(kite, lookback_days=120)
        
        # Get tokens for all symbols (NSE for spot, use name as token key)
        instruments = kite.instruments("NSE")
        inst_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
        tokens = {}
        for p in pairs:
            if p['stock_y'] in inst_map:
                tokens[p['stock_y']] = inst_map[p['stock_y']]
            if p['stock_x'] in inst_map:
                tokens[p['stock_x']] = inst_map[p['stock_x']]
        cache.set_tokens(tokens)
        
        for p in pairs:
            sym_y = p['stock_y']
            sym_x = p['stock_x']
            beta = p['beta']
            intercept = p['intercept']
            
            print(f"\n{'='*80}")
            print(f"   PAIR: {sym_y} vs {sym_x}")
            print(f"   Sector: {p.get('sector', 'N/A')} | Config Beta: {beta:.4f} | Config Intercept: {intercept:.2f}")
            print(f"{'='*80}")
            
            # Get historical data
            data_y = cache.get_data(sym_y)
            data_x = cache.get_data(sym_x)
            
            if data_y.empty or data_x.empty:
                print(f"   ⚠️ No data available for {sym_y} or {sym_x}")
                continue
            
            # Create bot and print stats
            bot = StatArbBot()
            bot.beta = beta
            bot.intercept = intercept
            bot.print_regression_stats(data_y, data_x, sym_y, sym_x)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Run 'python cli.py login' first to authenticate")

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


def cmd_download_backtest_spot(_args):
    """Download 750 days spot data for backtesting (research-backed duration)."""
    import json
    from datetime import datetime, timedelta
    
    print("\n--- ⬇️ DOWNLOADING BACKTEST SPOT DATA (750 days) ---")
    
    # Load validated pairs
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
    
    print(f"   📊 {len(symbols)} unique symbols from {len(pairs)} pairs")
    print(f"   📅 Duration: {config.BACKTEST_SPOT_DAYS} days")
    print(f"   📁 Output: {config.BACKTEST_SPOT_DIR}")
    
    # Date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=config.BACKTEST_SPOT_DAYS)).strftime("%Y-%m-%d")
    
    # Download
    download_historical_data(symbols, start_date, end_date, interval="day",
                             output_dir=config.BACKTEST_SPOT_DIR)
    
    print(f"\n✅ Backtest spot data saved to {config.BACKTEST_SPOT_DIR}")


def cmd_download_backtest_futures(_args):
    """Download futures data for backtesting (saves to historical/futures)."""
    from infrastructure.data.futures_utils import get_futures_details, download_futures_historical
    from infrastructure.broker.kite_auth import get_kite
    import pandas as pd
    import json
    from datetime import datetime, timedelta
    
    print("\n--- ⬇️ DOWNLOADING BACKTEST FUTURES DATA ---")
    
    # Load validated pairs
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
    
    print(f"   📊 {len(symbols)} unique symbols from {len(pairs)} pairs")
    print(f"   📁 Output: {config.BACKTEST_FUTURES_DIR}")
    
    # Connect to Kite
    try:
        kite = get_kite()
    except Exception as e:
        print(f"❌ Failed to connect to Kite: {e}")
        print("   Run 'python cli.py login' first")
        return
    
    # Date range (2 years for futures)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    print(f"   📅 Date range: {start_date} to {end_date}")
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
                symbol, start_date, end_date, "day",
                continuous=True, kite=kite
            )
            
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                filename = f"{details['symbol']}_day.csv"
                filepath = os.path.join(config.BACKTEST_FUTURES_DIR, filename)
                df.to_csv(filepath, index=False)
                print(f"✅ {len(df)} rows → {filename}")
                success += 1
            else:
                print("❌ No data")
                failed.append(symbol)
                
        except Exception as e:
            print(f"❌ {str(e)[:30]}")
            failed.append(symbol)
    
    print(f"\n✅ Downloaded: {success}/{len(symbols)} symbols to {config.BACKTEST_FUTURES_DIR}")
    if failed:
        print(f"❌ Failed: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")


def cmd_download_backtest_all(_args):
    """Download both spot (750 days) and futures data for backtesting."""
    print("\n=== DOWNLOADING ALL BACKTEST DATA ===\n")
    print("Step 1/2: Downloading spot data...")
    cmd_download_backtest_spot(_args)
    print("\nStep 2/2: Downloading futures data...")
    cmd_download_backtest_futures(_args)
    print("\n✅ All backtest data downloaded!")

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
    subparsers.add_parser("fetch_universe", help="0. Fetch Futures Symbols from Kite").set_defaults(func=cmd_fetch_universe)
    subparsers.add_parser("scan_fundamental", help="1. Financial Health Check").set_defaults(func=cmd_scan_fundamental)
    subparsers.add_parser("sector_analysis", help="2. Sector Deep Dive").set_defaults(func=cmd_sector_analysis)
    subparsers.add_parser("scan_pairs", help="3. Find Cointegrated Pairs (Method 2)").set_defaults(func=cmd_scan_pairs)
    subparsers.add_parser("backtest_pairs", help="4. Run Pro Backtest with Guardian").set_defaults(func=cmd_backtest_pairs)
    subparsers.add_parser("ai_advisor", help="5. AI Advisor - System Oversight").set_defaults(func=cmd_ai_advisor)
    
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
    
    # NEW: Backtest data download commands
    subparsers.add_parser("download_backtest_spot", help="Download 750 days spot data for backtesting").set_defaults(func=cmd_download_backtest_spot)
    subparsers.add_parser("download_backtest_futures", help="Download futures data for backtesting").set_defaults(func=cmd_download_backtest_futures)
    subparsers.add_parser("download_backtest_all", help="Download both spot (750d) + futures for backtesting").set_defaults(func=cmd_download_backtest_all)
    
    # 4. TRADING FLOOR (Updated for v2.0)
    p_eng = subparsers.add_parser("engine", help="Run Trading Engine v2.0")
    p_eng.add_argument("--mode", choices=["PAPER", "LIVE"], default="PAPER")
    p_eng.add_argument("--websocket", action="store_true", help="Enable real-time WebSocket updates")
    p_eng.set_defaults(func=cmd_engine)
    
    # 4b. LIVE POSITION TRACKER
    p_pos = subparsers.add_parser("positions", help="Live position tracker with real-time P&L")
    p_pos.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds (default: 5)")
    p_pos.add_argument("--websocket", action="store_true", help="Use WebSocket for real-time streaming")
    p_pos.set_defaults(func=cmd_positions)
    
    # 6. REPORTING
    subparsers.add_parser("pair-report", help="Generate pair data report (Z-scores, ADF)").set_defaults(func=cmd_pair_report)
    
    p_analytics = subparsers.add_parser("analytics", help="Generate trade analytics report")
    p_analytics.add_argument("--days", type=int, default=30, help="Days to analyze (default: 30)")
    p_analytics.set_defaults(func=cmd_analytics)
    
    subparsers.add_parser("daily-report", help="Generate daily P&L report").set_defaults(func=cmd_daily_report)
    subparsers.add_parser("pair-stats", help="Display regression statistics for configured pairs").set_defaults(func=cmd_pair_stats)
    
    p_ai = subparsers.add_parser("ai-analysis", help="AI-powered post-trade analysis (Gemini)")
    p_ai.add_argument("--days", type=int, default=30, help="Days to analyze (default: 30)")
    p_ai.set_defaults(func=cmd_ai_analysis)
    
    subparsers.add_parser("analyze_backtest", help="Analyze backtest results with AI").set_defaults(func=cmd_analyze_backtest)
    
    subparsers.add_parser("news-scan", help="Scan positions for corporate actions & critical news").set_defaults(func=cmd_news_scan)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

