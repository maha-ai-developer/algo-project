import sys
import os

# Ensure root path is added so we can import infrastructure.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import infrastructure.config as config
from infrastructure.broker.kite_auth import get_kite

def round_to_tick(price, tick_size=0.05):
    """
    Rounds a price to the nearest valid tick size (e.g. 0.05) and cleans floats.
    """
    if price is None: return None
    # 1. Round to nearest tick
    value = round(price / tick_size) * tick_size
    # 2. Force 2 decimal places to remove 3918.6000000004 artifacts
    return round(value, 2)

def place_order(symbol, side, quantity, price=0, trigger_price=0, order_type="LIMIT", tag="algo_trade"):
    """
    Places an order via Kite Connect.
    
    Args:
        symbol (str): Trading Symbol (e.g. "INFY")
        side (str): "BUY" or "SELL"  <-- UPDATED NAME TO MATCH EXECUTION.PY
        quantity (int): Number of shares
        price (float): Limit Price (Required for LIMIT/SL)
        trigger_price (float): Trigger Price (Required for SL/SL-M)
        order_type (str): "MARKET", "LIMIT", "SL", "SL-M"
        tag (str): Order tag for identifying algo trades
    """
    try:
        kite = get_kite()
        
        # Map Side to Kite Constant
        tx_type = kite.TRANSACTION_TYPE_BUY if side == "BUY" else kite.TRANSACTION_TYPE_SELL
        
        # Map Order Type
        if order_type == "SL":
            kite_order_type = kite.ORDER_TYPE_SL
        elif order_type == "SL-M":
            kite_order_type = kite.ORDER_TYPE_SLM
        elif order_type == "MARKET":
            kite_order_type = kite.ORDER_TYPE_MARKET
        else:
            kite_order_type = kite.ORDER_TYPE_LIMIT

        # Round Prices (Safety Check)
        limit_price = round_to_tick(price) if order_type in ["LIMIT", "SL"] else None
        trig_price = round_to_tick(trigger_price) if trigger_price and trigger_price > 0 else None

        print(f"📞 Sending {side}: {symbol} Qty {quantity} | Type: {order_type} | Price: {limit_price} | Trig: {trig_price}")

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

def cancel_order(order_id):
    """
    Cancels a specific order ID.
    """
    try:
        kite = get_kite()
        kite.cancel_order(
            variety=kite.VARIETY_REGULAR,
            order_id=order_id
        )
        print(f"🗑️ Cancelled Order: {order_id}")
        return True
    except Exception as e:
        print(f"❌ Cancel Failed: {e}")
        return False

def modify_order(order_id, new_price, new_trigger_price, order_type="SL"):
    """
    Modifies an open order to a new price (Trailing Stop Logic).
    """
    try:
        kite = get_kite()
        
        # Rounding for safety
        if new_price: new_price = round_to_tick(new_price)
        if new_trigger_price: new_trigger_price = round_to_tick(new_trigger_price)

        print(f"♻️ MODIFYING Order {order_id} -> Price: {new_price} | Trig: {new_trigger_price}")

        kite.modify_order(
            variety=kite.VARIETY_REGULAR,
            order_id=order_id,
            order_type=kite.ORDER_TYPE_SL if order_type == "SL" else kite.ORDER_TYPE_SLM,
            price=new_price,
            trigger_price=new_trigger_price
        )
        return True
    except Exception as e:
        print(f"❌ Modify Failed: {e}")
        return False
