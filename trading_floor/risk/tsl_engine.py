# risk/tsl_engine.py
from typing import Optional


def compute_trailing_sl(entry_price: float,
                        highest_price: float,
                        trail_pct: float = 0.5) -> Optional[float]:
    """
    Simple trailing SL:
      - For a long:
        trail_stop = highest_price * (1 - trail_pct/100)
      - trail_pct default: 0.5% below highest
    """
    if highest_price <= 0:
        return None
    tsl = highest_price * (1 - trail_pct / 100.0)
    if tsl <= 0:
        return None
    return round(tsl, 2)
