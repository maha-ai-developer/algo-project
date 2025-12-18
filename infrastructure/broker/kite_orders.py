import sys
import os
import logging

# Ensure root path is added
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import infrastructure.config as config
from infrastructure.broker.kite_auth import get_kite

# GLOBAL CACHE: Stores instrument metadata (tick sizes) to avoid repeated API calls
# Structure: {'INFY': 0.05, 'LT': 0.10, 'RELIANCE': 0.05, ...}
_INSTRUMENT_CACHE = {}

def load_instrument_master():
    """
    Fetches the complete list of NSE instruments from Kite.
    Populates _INSTRUMENT_CACHE with accurate tick sizes for all stocks.
    This runs only once per session.
    """
    global _INSTRUMENT_CACHE
    if _INSTRUMENT_CACHE:
        return  # Cache is already loaded, skip fetch

    print("   ⬇️ Fetching Instrument Master to calibrate Tick Sizes...")
    try:
        kite = get_kite()
        # Fetch all instruments for NSE
        instruments = kite.instruments("NSE")
        
        # Populate the dictionary for O(1) fast lookup
        for inst in instruments:
            symbol = inst['tradingsymbol']
            _INSTRUMENT_CACHE[symbol] = inst['tick_size']
            
        print(f"   ✅ Instrument Master Loaded ({len(_INSTRUMENT_CACHE)} symbols processed)")
        
    except Exception as e:
        print(f"   ⚠️ Critical: Failed to load Instrument Master. Error: {e}")
        # We do not exit here; individual lookups will fallback to default if cache is empty.

def get_dynamic_tick_size(symbol):
    """
    Retrieves the exact tick size for a symbol from the loaded cache.
    """
    # 1. Ensure Cache is loaded (Lazy Loading)
    if not _INSTRUMENT_CACHE:
        load_instrument_master()

    # 2. Look up the symbol
    if symbol in _INSTRUMENT_CACHE:
        return _INSTRUMENT_CACHE[symbol]

    # 3. Safety Fallback (Log warning but proceed with standard 0.05)
    # This only happens if the symbol is invalid or not in the NSE segment
    print(f"   ⚠️ Warning: Symbol '{symbol}' not found in Master. Defaulting to 0.05 tick.")
    return 0.05

def round_to_tick(price, tick_size=0.05):
    """
    Rounds a price to the nearest valid tick size.
    Example: If tick is 0.10, price 4059.15 -> 4059.10
    """
    if price is None: return None
    value = round(price / tick_size) * tick_size
    return round(value, 2)

def place_order(symbol, side, quantity, price=0, trigger_price=0, order_type="LIMIT", tag="algo_trade"):
    """
    Places an order via Kite Connect using 100% dynamic tick size resolution.
    """
    try:
        kite = get_kite()
        
        # 1. DYNAMIC TICK FETCH (No hardcoding)
        tick_size = get_dynamic_tick_size(symbol)
        
        # 2. Map Side
        tx_type = kite.TRANSACTION_TYPE_BUY if side == "BUY" else kite.TRANSACTION_TYPE_SELL
        
        # 3. Map Order Type
        if order_type == "SL": kite_order_type = kite.ORDER_TYPE_SL
        elif order_type == "SL-M": kite_order_type = kite.ORDER_TYPE_SLM
        elif order_type == "MARKET": kite_order_type = kite.ORDER_TYPE_MARKET
        else: kite_order_type = kite.ORDER_TYPE_LIMIT

        # 4. Round Prices
        # This guarantees the price is compliant with the exchange's specific rules for THIS stock
        limit_price = round_to_tick(price, tick_size) if order_type in ["LIMIT", "SL"] else None
        trig_price = round_to_tick(trigger_price, tick_size) if trigger_price and trigger_price > 0 else None

        print(f"📞 Sending {side}: {symbol} Qty {quantity} | Type: {order_type} | Price: {limit_price} | Tick: {tick_size}")

        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=tx_type,
            quantity=quantity,
            product=kite.PRODUCT_MIS,
            order_type=kite_order_type,
            price=limit_price,
            trigger_price=trig_price,
            tag=tag
        )
        
        print(f"✅ Order Placed! ID: {order_id}")
        return order_id

    except Exception as e:
        print(f"❌ Order Placement Failed: {e}")
        return None
