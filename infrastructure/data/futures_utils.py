"""
Futures Utilities Module v2.0 - Kite API Integrated

Uses actual Kite Connect API for:
- Real-time lot sizes from kite.instruments("NFO")
- Actual expiry dates
- Live margin calculations via kite.order_margins()
- Continuous data support for backtesting
"""

import os
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import calendar

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import infrastructure.config as config


# ============================================================
# INSTRUMENT CACHE
# ============================================================

class InstrumentCache:
    """
    Caches NFO instruments from Kite API to avoid repeated calls.
    Cache is refreshed daily (instruments change rarely within a day).
    """
    
    _stale_warning_shown = False  # Class-level flag to prevent spam
    
    def __init__(self, cache_file: Optional[str] = None):
        self.cache_file = cache_file or os.path.join(config.CACHE_DIR, "nfo_instruments.json")
        self._instruments: List[Dict] = []
        self._cache_date: Optional[date] = None
        self._loaded = False
    
    def _load_from_file(self) -> bool:
        """Load cached instruments from file."""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            cache_date = datetime.fromisoformat(data.get('_cache_date', '2000-01-01')).date()
            
            # Cache valid for 1 day
            if cache_date != date.today():
                return False
            
            self._instruments = data.get('instruments', [])
            self._cache_date = cache_date
            self._loaded = True
            return True
            
        except (json.JSONDecodeError, ValueError):
            return False
    
    def _save_to_file(self):
        """Save instruments to cache file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'instruments': self._instruments,
                    '_cache_date': date.today().isoformat()
                }, f)
        except Exception as e:
            print(f"   ⚠️ Failed to cache instruments: {e}")
    
    def fetch_from_kite(self, kite) -> List[Dict]:
        """
        Fetch NFO instruments from Kite API.
        
        Args:
            kite: Authenticated KiteConnect instance
        
        Returns:
            List of instrument dictionaries
        """
        try:
            print("   📥 Downloading NFO instrument master from Kite...")
            instruments = kite.instruments("NFO")
            
            # Convert date objects to strings for JSON serialization
            for inst in instruments:
                if 'expiry' in inst and inst['expiry']:
                    inst['expiry'] = inst['expiry'].isoformat() if hasattr(inst['expiry'], 'isoformat') else str(inst['expiry'])
            
            self._instruments = instruments
            self._cache_date = date.today()
            self._loaded = True
            
            self._save_to_file()
            print(f"   ✅ Loaded {len(instruments)} NFO instruments")
            
            return instruments
            
        except Exception as e:
            print(f"   ❌ Failed to fetch instruments: {e}")
            # Try loading from cache even if expired
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                self._instruments = data.get('instruments', [])
                self._loaded = True
                if not InstrumentCache._stale_warning_shown:
                    print(f"   ⚠️ Using stale cache ({len(self._instruments)} instruments)")
                    InstrumentCache._stale_warning_shown = True
                return self._instruments
            return []
    
    def get_instruments(self, kite=None) -> List[Dict]:
        """Get instruments, loading from cache or API as needed."""
        if self._loaded:
            return self._instruments
        
        if self._load_from_file():
            return self._instruments
        
        if kite:
            return self.fetch_from_kite(kite)
        
        # Fallback: try loading from expired cache
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            self._instruments = data.get('instruments', [])
            self._loaded = True
            if not InstrumentCache._stale_warning_shown:
                print(f"   ⚠️ Using stale cache ({len(self._instruments)} instruments)")
                InstrumentCache._stale_warning_shown = True
            return self._instruments
        
        return []


# Global cache instance
_instrument_cache = InstrumentCache()


# ============================================================
# FUTURES LOOKUP (From Kite Instruments)
# ============================================================

def get_futures_details(symbol_root: str, kite=None) -> Optional[Dict]:
    """
    Get current month futures details from Kite instruments.
    
    Args:
        symbol_root: e.g., "SBIN", "RELIANCE"
        kite: Optional KiteConnect instance (for live refresh)
    
    Returns:
        Dict with: Symbol, lot_size, expiry, instrument_token
    """
    instruments = _instrument_cache.get_instruments(kite)
    
    if not instruments:
        return None
    
    # Filter for Futures of the specific symbol
    today = date.today()
    
    futures = [
        i for i in instruments 
        if i.get('name') == symbol_root.upper() and i.get('instrument_type') == 'FUT'
    ]
    
    if not futures:
        return None
    
    # Sort by Expiry to find the "Current Month" (Nearest Expiry)
    def parse_expiry(exp):
        if isinstance(exp, str):
            return datetime.fromisoformat(exp).date()
        return exp
    
    valid_futures = [
        f for f in futures 
        if f.get('expiry') and parse_expiry(f['expiry']) >= today
    ]
    
    if not valid_futures:
        return None
    
    valid_futures.sort(key=lambda x: parse_expiry(x['expiry']))
    current_month = valid_futures[0]
    
    return {
        "symbol": current_month['tradingsymbol'],
        "lot_size": current_month['lot_size'],
        "expiry": current_month['expiry'],
        "instrument_token": current_month['instrument_token'],
        "exchange": current_month.get('exchange', 'NFO'),
        "tick_size": current_month.get('tick_size', 0.05)
    }


def get_lot_size(symbol: str, kite=None) -> int:
    """
    Get lot size for a symbol from Kite instruments.
    
    Falls back to hardcoded defaults if Kite data unavailable.
    """
    # Try to get from Kite instruments first
    details = get_futures_details(symbol, kite)
    if details:
        return details['lot_size']
    
    # Fallback: hardcoded defaults (Updated Jan 2025)
    DEFAULT_LOT_SIZES = {
        "SBIN": 1500, "HDFCBANK": 550, "ICICIBANK": 1400, "KOTAKBANK": 400,
        "AXISBANK": 1200, "RELIANCE": 250, "TCS": 150, "INFY": 400,
        "MARUTI": 100, "TATAMOTORS": 1400, "M&M": 350, "BAJAJ-AUTO": 125,
        "EICHERMOT": 150, "TATASTEEL": 5500, "JSWSTEEL": 1000,
        "SUNPHARMA": 350, "HINDUNILVR": 300, "ITC": 1600,
        "BHARTIARTL": 950, "LT": 150, "TITAN": 175,
        "BAJFINANCE": 125, "NIFTY": 50, "BANKNIFTY": 15,
    }
    
    base_symbol = symbol.upper()
    # Remove futures suffix if present
    for suffix in ["FUT", "25", "24"]:
        base_symbol = base_symbol.replace(suffix, "")
    for month in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", 
                  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]:
        base_symbol = base_symbol.replace(month, "")
    
    return DEFAULT_LOT_SIZES.get(base_symbol, 1)


def get_all_expiries(symbol_root: str, kite=None) -> List[Dict]:
    """
    Get all available expiries for a symbol (Near, Next, Far).
    
    Returns:
        List of dicts with symbol, expiry, lot_size, token
    """
    instruments = _instrument_cache.get_instruments(kite)
    
    if not instruments:
        return []
    
    today = date.today()
    
    futures = [
        i for i in instruments 
        if i.get('name') == symbol_root.upper() and i.get('instrument_type') == 'FUT'
    ]
    
    def parse_expiry(exp):
        if isinstance(exp, str):
            return datetime.fromisoformat(exp).date()
        return exp
    
    valid_futures = [
        f for f in futures 
        if f.get('expiry') and parse_expiry(f['expiry']) >= today
    ]
    
    valid_futures.sort(key=lambda x: parse_expiry(x['expiry']))
    
    return [
        {
            "symbol": f['tradingsymbol'],
            "expiry": f['expiry'],
            "lot_size": f['lot_size'],
            "instrument_token": f['instrument_token'],
            "month": "NEAR" if i == 0 else ("NEXT" if i == 1 else "FAR")
        }
        for i, f in enumerate(valid_futures[:3])
    ]


# ============================================================
# MARGIN CALCULATIONS (Via Kite API)
# ============================================================

def get_margin_required(symbol: str, quantity: int, transaction_type: str = "BUY", kite=None) -> Optional[Dict]:
    """
    Get actual margin required from Kite API.
    
    Args:
        symbol: Futures trading symbol (e.g., "SBIN25JANFUT")
        quantity: Number of shares (lot_size * lots)
        transaction_type: "BUY" or "SELL"
        kite: Authenticated KiteConnect instance
    
    Returns:
        Dict with margin details or None
    """
    if not kite:
        # Fallback: estimate 15% margin
        details = get_futures_details(symbol.replace("FUT", "").rstrip("0123456789")[:10])
        if details:
            # Rough estimate: current price unknown, return percentage
            return {
                "total": quantity * 100 * 0.15,  # Rough estimate
                "span": quantity * 100 * 0.10,
                "exposure": quantity * 100 * 0.05,
                "estimated": True
            }
        return None
    
    try:
        order_params = {
            "exchange": "NFO",
            "tradingsymbol": symbol,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "product": "NRML",
            "order_type": "MARKET"
        }
        
        margins = kite.order_margins([order_params])
        
        if margins and len(margins) > 0:
            m = margins[0]
            return {
                "total": m.get('total', 0),
                "span": m.get('span', 0),
                "exposure": m.get('exposure', 0),
                "additional": m.get('additional', 0),
                "estimated": False
            }
    except Exception as e:
        print(f"   ⚠️ Margin API call failed: {e}")
    
    return None


def calculate_margin_required(symbol: str, price: float, lots: int = 1, kite=None) -> float:
    """
    Calculate margin required for a position.
    
    Uses Kite API if available, otherwise estimates at 15%.
    """
    lot_size = get_lot_size(symbol, kite)
    quantity = lot_size * lots
    
    # Try Kite API first
    if kite:
        margin_info = get_margin_required(symbol, quantity, "BUY", kite)
        if margin_info and not margin_info.get('estimated', True):
            return margin_info['total']
    
    # Fallback: 15% margin estimate
    contract_value = price * quantity
    return contract_value * 0.15


# ============================================================
# HISTORICAL DATA HELPERS
# ============================================================

def download_futures_historical(symbol_root: str, from_date: str, to_date: str, 
                                 interval: str = "day", continuous: bool = True, 
                                 kite=None) -> Optional[List[Dict]]:
    """
    Download historical data for futures.
    
    Args:
        symbol_root: e.g., "SBIN"
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        interval: "minute", "day", etc.
        continuous: If True, use continuous data (handles rollover)
        kite: Authenticated KiteConnect instance
    
    Returns:
        List of OHLC data or None
    """
    if not kite:
        print("   ❌ Kite instance required for historical data")
        return None
    
    # Get current futures details
    details = get_futures_details(symbol_root, kite)
    if not details:
        print(f"   ❌ No futures found for {symbol_root}")
        return None
    
    try:
        token = details['instrument_token']
        
        data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=continuous,
            oi=True  # Include Open Interest
        )
        
        return data
        
    except Exception as e:
        print(f"   ❌ Failed to download historical data: {e}")
        return None


# ============================================================
# SYMBOL MAPPING (Fallback for offline use)
# ============================================================

def get_expiry_date(year: int, month: int) -> date:
    """
    Get last Thursday of the month (F&O expiry).
    Used as fallback when Kite data unavailable.
    """
    last_day = calendar.monthrange(year, month)[1]
    
    for day in range(last_day, 0, -1):
        if date(year, month, day).weekday() == 3:  # Thursday
            return date(year, month, day)
    
    return date(year, month, last_day)


def get_futures_symbol(spot_symbol: str, expiry_date: date = None, expiry_str: str = None) -> str:
    """
    Convert spot symbol to futures symbol.
    
    Args:
        spot_symbol: e.g., "SBIN"
        expiry_date: Expiry date object (optional)
        expiry_str: Expiry string like "2026-01" for manual override (optional)
    
    Returns:
        Futures symbol, e.g., "SBIN26JANFUT"
    """
    # Priority: expiry_str > expiry_date > auto-detect
    if expiry_str:
        # Parse "2026-01" format
        try:
            year, month = map(int, expiry_str.split('-'))
            expiry = get_expiry_date(year, month)
        except:
            print(f"   ⚠️ Invalid expiry format '{expiry_str}', using auto-detect")
            expiry_str = None
    
    if not expiry_str:
        if expiry_date is None:
            today = date.today()
            expiry = get_expiry_date(today.year, today.month)
            if today > expiry:
                # Past expiry, use next month
                next_month = today.month + 1
                next_year = today.year
                if next_month > 12:
                    next_month = 1
                    next_year += 1
                expiry = get_expiry_date(next_year, next_month)
        else:
            expiry = expiry_date
    
    year_short = expiry.strftime("%y")
    month_short = expiry.strftime("%b").upper()
    
    return f"{spot_symbol.upper()}{year_short}{month_short}FUT"


def get_current_month_future(spot_symbol: str, kite=None) -> str:
    """
    Get current month's futures symbol.
    
    Prefers Kite API data, falls back to calculation.
    """
    details = get_futures_details(spot_symbol, kite)
    if details:
        return details['symbol']
    
    return get_futures_symbol(spot_symbol)


# ============================================================
# CONTRACT INFO SUMMARY
# ============================================================

def get_contract_info(symbol: str, price: float = 0, kite=None) -> Dict:
    """Get complete contract information for a symbol."""
    details = get_futures_details(symbol, kite)
    
    if details:
        lot_size = details['lot_size']
        expiry = details['expiry']
        futures_symbol = details['symbol']
        token = details['instrument_token']
    else:
        lot_size = get_lot_size(symbol)
        expiry = get_expiry_date(date.today().year, date.today().month).isoformat()
        futures_symbol = get_futures_symbol(symbol)
        token = None
    
    contract_value = price * lot_size if price > 0 else 0
    margin_required = contract_value * 0.15
    
    return {
        "spot_symbol": symbol,
        "futures_symbol": futures_symbol,
        "lot_size": lot_size,
        "expiry": expiry,
        "instrument_token": token,
        "price": price,
        "contract_value": round(contract_value, 2),
        "margin_pct": "15%",
        "margin_required": round(margin_required, 2),
        "data_source": "KITE_API" if details else "FALLBACK"
    }


# ============================================================
# EXPIRY MANAGEMENT (NEW - Checklist Phase 8)
# ============================================================

def days_to_expiry(symbol: str, kite=None) -> int:
    """
    Calculate trading days until expiry.
    
    Args:
        symbol: Spot or futures symbol (e.g., "SBIN" or "SBIN25JANFUT")
        kite: Optional KiteConnect instance for live refresh
        
    Returns:
        Number of calendar days to expiry (0 if expired, -1 if error)
    """
    # Extract base symbol if futures symbol provided
    base_symbol = symbol.upper()
    for suffix in ["FUT", "25", "24", "26"]:
        base_symbol = base_symbol.replace(suffix, "")
    for month in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", 
                  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]:
        base_symbol = base_symbol.replace(month, "")
    
    details = get_futures_details(base_symbol, kite)
    
    if not details or not details.get('expiry'):
        # Fallback: calculate from current month
        today = date.today()
        expiry = get_expiry_date(today.year, today.month)
        if today > expiry:
            # Use next month
            next_month = today.month + 1
            next_year = today.year
            if next_month > 12:
                next_month = 1
                next_year += 1
            expiry = get_expiry_date(next_year, next_month)
    else:
        expiry_str = details['expiry']
        if isinstance(expiry_str, str):
            expiry = datetime.fromisoformat(expiry_str).date()
        else:
            expiry = expiry_str
    
    days = (expiry - date.today()).days
    return max(0, days)


class RolloverManager:
    """
    Manages F&O contract rollover - NEW Checklist Phase 8.
    
    - Warns at 3 days to expiry
    - Provides next month symbol
    - Executes rollover (close current, open next)
    """
    
    ROLLOVER_WARNING_DAYS = 3  # Warn when expiry within 3 days
    
    def __init__(self, kite=None):
        """
        Initialize RolloverManager.
        
        Args:
            kite: Optional KiteConnect instance for live data
        """
        self.kite = kite
    
    def check_expiry_proximity(self, symbol: str) -> Tuple[bool, int, str]:
        """
        Check if position is approaching expiry.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (needs_rollover_warning, days_remaining, message)
        """
        days = days_to_expiry(symbol, self.kite)
        
        if days <= 0:
            return True, 0, f"{symbol}: EXPIRED or expiry TODAY - URGENT ROLLOVER"
        elif days <= self.ROLLOVER_WARNING_DAYS:
            return True, days, f"{symbol}: {days} days to expiry - Consider rollover"
        else:
            return False, days, f"{symbol}: {days} days to expiry - OK"
    
    def get_next_month_symbol(self, spot_symbol: str) -> str:
        """
        Get next month's futures symbol for rollover.
        
        Args:
            spot_symbol: Base stock symbol (e.g., "SBIN")
            
        Returns:
            Next month futures symbol (e.g., "SBIN25FEBFUT")
        """
        today = date.today()
        current_expiry = get_expiry_date(today.year, today.month)
        
        # Determine next month
        if today <= current_expiry:
            # Still in current month, next is +1 month
            next_month = today.month + 1
            next_year = today.year
        else:
            # Past current expiry, next is +2 months
            next_month = today.month + 2
            next_year = today.year
        
        # Handle year rollover
        if next_month > 12:
            next_month -= 12
            next_year += 1
        
        next_expiry = get_expiry_date(next_year, next_month)
        return get_futures_symbol(spot_symbol, expiry_date=next_expiry)
    
    def should_block_entry(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if new entries should be blocked due to imminent expiry.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (should_block, reason)
        """
        days = days_to_expiry(symbol, self.kite)
        
        if days <= self.ROLLOVER_WARNING_DAYS:
            return True, f"Expiry in {days} days - Use next month contract"
        return False, ""
    
    def get_rollover_plan(self, spot_symbol: str, 
                          current_qty: int, 
                          current_side: str) -> Dict:
        """
        Generate rollover execution plan.
        
        Args:
            spot_symbol: Base stock symbol
            current_qty: Current position quantity
            current_side: Current position side ("BUY" or "SELL")
            
        Returns:
            Dict with close and open orders
        """
        current_future = get_current_month_future(spot_symbol, self.kite)
        next_future = self.get_next_month_symbol(spot_symbol)
        
        # To close: opposite of current side
        close_side = "SELL" if current_side == "BUY" else "BUY"
        
        return {
            "close_order": {
                "symbol": current_future,
                "side": close_side,
                "quantity": current_qty,
                "action": "CLOSE_CURRENT_MONTH"
            },
            "open_order": {
                "symbol": next_future,
                "side": current_side,  # Same direction in new month
                "quantity": current_qty,
                "action": "OPEN_NEXT_MONTH"
            }
        }


# ============================================================
# REFRESH CACHE
# ============================================================

def refresh_instrument_cache(kite) -> int:
    """
    Force refresh the instrument cache from Kite API.
    
    Returns:
        Number of instruments loaded
    """
    instruments = _instrument_cache.fetch_from_kite(kite)
    return len(instruments)

