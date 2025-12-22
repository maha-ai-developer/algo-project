import sys
import os
import logging

# Ensure root path is added
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import infrastructure.config as config
from infrastructure.broker.kite_auth import get_kite

# GLOBAL CACHE: Stores instrument metadata
_INSTRUMENT_CACHE = {}

def load_instrument_master():
    """
    Fetches the complete list of NSE instruments from Kite.
    Populates _INSTRUMENT_CACHE with accurate tick sizes.
    """
    global _INSTRUMENT_CACHE
    if _INSTRUMENT_CACHE:
        return

    print("   ⬇️ Fetching Instrument Master to calibrate Tick Sizes...")
    try:
        kite = get_kite()
        instruments = kite.instruments("NSE")
        
        for inst in instruments:
            symbol = inst['tradingsymbol']
            _INSTRUMENT_CACHE[symbol] = inst['tick_size']
            
        print(f"   ✅ Instrument Master Loaded ({len(_INSTRUMENT_CACHE)} symbols processed)")
        
    except Exception as e:
        print(f"   ⚠️ Critical: Failed to load Instrument Master. Error: {e}")

def get_dynamic_tick_size(symbol):
    """Retrieves exact tick size."""
    if not _INSTRUMENT_CACHE:
        load_instrument_master()

    if symbol in _INSTRUMENT_CACHE:
        return _INSTRUMENT_CACHE[symbol]

    print(f"   ⚠️ Warning: Symbol '{symbol}' not found in Master. Defaulting to 0.05 tick.")
    return 0.05

def round_to_tick(price, tick_size=0.05):
    """Rounds a price to the nearest valid tick size."""
    if price is None: return None
    value = round(price / tick_size) * tick_size
    return round(value, 2)

# --- UPDATED FUNCTION SIGNATURE ---
def place_order(symbol, side, quantity, price=0, trigger_price=0, order_type="LIMIT", product="MIS", tag="algo_trade"):
    """
    Places an order via Kite Connect.
    
    CRITICAL UPDATE: Added 'product' parameter.
    - Use "MIS" for Intraday.
    - Use "NRML" for Futures Overnight.
    - Use "CNC" for Equity Delivery Overnight.
    """
    try:
        kite = get_kite()
        
        # 1. DYNAMIC TICK FETCH
        tick_size = get_dynamic_tick_size(symbol)
        
        # 2. Map Side
        tx_type = kite.TRANSACTION_TYPE_BUY if side == "BUY" else kite.TRANSACTION_TYPE_SELL
        
        # 3. Map Order Type
        if order_type == "SL": kite_order_type = kite.ORDER_TYPE_SL
        elif order_type == "SL-M": kite_order_type = kite.ORDER_TYPE_SLM
        elif order_type == "MARKET": kite_order_type = kite.ORDER_TYPE_MARKET
        else: kite_order_type = kite.ORDER_TYPE_LIMIT

        # 4. Map Product Type (The Fix)
        if product == "CNC": kite_product = kite.PRODUCT_CNC
        elif product == "NRML": kite_product = kite.PRODUCT_NRML
        else: kite_product = kite.PRODUCT_MIS

        # 5. Round Prices
        limit_price = round_to_tick(price, tick_size) if order_type in ["LIMIT", "SL"] else None
        trig_price = round_to_tick(trigger_price, tick_size) if trigger_price and trigger_price > 0 else None

        print(f"📞 Sending {side}: {symbol} Qty {quantity} | Type: {order_type} | Product: {product}")

        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=tx_type,
            quantity=quantity,
            product=kite_product,  # <--- USING VARIABLE NOW
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
