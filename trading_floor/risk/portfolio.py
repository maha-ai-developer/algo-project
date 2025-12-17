# risk/portfolio.py

from typing import Tuple, Dict
from broker.kite_positions import fetch_account_snapshot

class PortfolioLimits:
    def __init__(self, max_open_trades=5, max_leverage=1.0):
        self.max_open_trades = max_open_trades
        self.max_leverage = max_leverage

class PortfolioManager:
    def __init__(self, limits: PortfolioLimits = None):
        self.limits = limits or PortfolioLimits()
        self.open_trades = {}  # symbol -> qty
        self.cached_equity = 16000.0  # Safe default

    def update_positions(self, positions: Dict[str, int]):
        self.open_trades = positions

    def _get_live_equity(self) -> float:
        try:
            # fetch_account_snapshot returns 4 values:
            # profile, margins, holdings, positions
            _, margins, _, _ = fetch_account_snapshot()
            
            if margins:
                val = float(margins.get("net", margins.get("equity", 0.0)))
                if val > 1.0:
                    self.cached_equity = val
                    return val
        except Exception:
            pass
        return self.cached_equity

    def update_snapshot(self):
        """
        Called by RiskMonitor to refresh Equity & Positions from Broker.
        """
        # 1. Update Equity (Internal cache)
        self._get_live_equity()

        # 2. Update Positions
        try:
            # We call fetch_account_snapshot AGAIN to get positions
            # returns: profile, margins, holdings, positions
            _, _, _, positions_data = fetch_account_snapshot()
            
            new_map = {}
            # Zerodha positions response has 'net' and 'day' lists
            if positions_data and "net" in positions_data:
                for p in positions_data["net"]:
                    qty = int(p.get("quantity", 0))
                    if qty != 0:
                        new_map[p["tradingsymbol"]] = qty
            
            self.open_trades = new_map
        except Exception as e:
            print(f"[Portfolio] Snapshot update failed: {e}")

    def can_open_trade(self, symbol: str, side: str, price: float, qty: int) -> Tuple[bool, str]:
        equity = self._get_live_equity()
        
        if equity <= 0:
            print("[Portfolio] Equity 0 detected. Using Emergency Fallback ₹10,000")
            equity = 10000.0

        trade_value = price * qty
        if trade_value > (equity * 5.0):
             return False, f"margin required {trade_value} > 5x equity {equity}"

        return True, "OK"
