#!/usr/bin/env python3
"""
Utility script to fetch & search instrument tokens manually.
Usage:
    python -m infrastructure.data.instrument_token_finder SBIN
    python -m infrastructure.data.instrument_token_finder --list
"""

import argparse
import sys
import os
from tabulate import tabulate

# Add root to path so we can import infrastructure
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from infrastructure.broker.kite_auth import get_kite

def fetch_instruments():
    """
    Fetch all instruments from Kite API.
    """
    kite = get_kite()
    print("[INFO] Fetching full instruments list from Kite...")
    instruments = kite.instruments()
    print(f"[INFO] Loaded {len(instruments)} instruments.")
    return instruments

def search_exact(symbol: str, exchange: str, instruments):
    symbol = symbol.upper()
    exchange = exchange.upper()
    return [
        ins for ins in instruments
        if ins["tradingsymbol"] == symbol and ins["exchange"] == exchange
    ]

def search_partial(query: str, instruments):
    query = query.upper()
    return [
        ins for ins in instruments
        if query in ins["tradingsymbol"]
    ][:20] # Limit to 20 results

def list_all_for_exchange(exchange: str, instruments):
    exchange = exchange.upper()
    return [
        ins for ins in instruments
        if ins["exchange"] == exchange
    ]

def print_results(results):
    if not results:
        print("No instruments found.")
        return

    print(
        tabulate(
            [
                [
                    r["instrument_token"],
                    r["exchange"],
                    r["tradingsymbol"],
                    r["name"],
                    r["lot_size"],
                    r["segment"],
                    r["instrument_type"]
                ]
                for r in results
            ],
            headers=[
                "Token", "Exchange", "Symbol", "Name", "Lot", "Segment", "Type"
            ],
            tablefmt="fancy_grid"
        )
    )

def main():
    parser = argparse.ArgumentParser(description="Zerodha Instrument Token Finder")
    parser.add_argument("symbol", nargs="?", help="Trading symbol to search (e.g., SBIN)")
    parser.add_argument("--exchange", default="NSE", help="Exchange (NSE/BSE/NFO/CDS)")
    parser.add_argument("--search", action="store_true", help="Enable partial search")
    parser.add_argument("--list", action="store_true", help="List all for exchange")

    args = parser.parse_args()

    try:
        instruments = fetch_instruments()

        if args.list:
            results = list_all_for_exchange(args.exchange, instruments)
            print_results(results)
        elif args.search and args.symbol:
            results = search_partial(args.symbol, instruments)
            print_results(results)
        elif args.symbol:
            results = search_exact(args.symbol, args.exchange, instruments)
            print_results(results)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
