"""
WebSocket Ticker Module - Optimization #6

Real-time price updates via Kite WebSocket instead of REST polling.
Provides sub-second price updates vs 60-second polling intervals.
"""

import threading
import time
from typing import Dict, Callable, Optional, List
from datetime import datetime


class RealtimeTicker:
    """
    WebSocket-based real-time price ticker using KiteTicker.
    
    Features:
    - Auto-reconnection on disconnect
    - Thread-safe price store
    - Callback-based price updates
    - Graceful shutdown
    """
    
    def __init__(self, api_key: str, access_token: str):
        """
        Args:
            api_key: Kite Connect API key
            access_token: Valid access token
        """
        self.api_key = api_key
        self.access_token = access_token
        
        # Thread-safe price store
        self._prices: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        
        # State
        self._connected = False
        self._running = False
        self._ticker = None
        self._thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_price_update: Optional[Callable] = None
        self._subscribed_tokens: List[int] = []
    
    def connect(self, tokens: List[int], on_price_update: Optional[Callable] = None):
        """
        Connect to WebSocket and subscribe to tokens.
        
        Args:
            tokens: List of instrument tokens to subscribe
            on_price_update: Callback function(symbol, price, timestamp)
        """
        try:
            from kiteconnect import KiteTicker
        except ImportError:
            print("   ⚠️ kiteconnect not installed. WebSocket disabled.")
            return False
        
        self._subscribed_tokens = tokens
        self._on_price_update = on_price_update
        
        # Create ticker instance
        self._ticker = KiteTicker(self.api_key, self.access_token)
        
        # Set up callbacks
        self._ticker.on_ticks = self._handle_ticks
        self._ticker.on_connect = self._handle_connect
        self._ticker.on_close = self._handle_close
        self._ticker.on_error = self._handle_error
        self._ticker.on_reconnect = self._handle_reconnect
        
        # Start in background thread
        self._running = True
        self._thread = threading.Thread(target=self._run_ticker, daemon=True)
        self._thread.start()
        
        print("   🔌 WebSocket ticker starting...")
        return True
    
    def _run_ticker(self):
        """Run ticker in background thread."""
        try:
            # Use threaded=True to avoid signal handling issues
            # The Twisted reactor signal error is non-fatal - WebSocket still works
            import warnings
            warnings.filterwarnings("ignore", message=".*signal.*")
            
            self._ticker.connect(threaded=True)
        except ValueError as e:
            # Signal error from Twisted is non-fatal, WebSocket still connects
            if "signal only works in main thread" in str(e):
                pass  # Ignored - WebSocket still functional
            else:
                print(f"   ❌ WebSocket error: {e}")
                self._connected = False
        except Exception as e:
            print(f"   ❌ WebSocket error: {e}")
            self._connected = False
    
    def _handle_connect(self, ws, response):
        """Called when WebSocket connects."""
        self._connected = True
        print(f"   ✅ WebSocket connected. Subscribing to {len(self._subscribed_tokens)} tokens...")
        
        # Subscribe to tokens
        if self._subscribed_tokens:
            self._ticker.subscribe(self._subscribed_tokens)
            # Set mode to full quotes for complete data
            self._ticker.set_mode(self._ticker.MODE_FULL, self._subscribed_tokens)
    
    def _handle_ticks(self, ws, ticks):
        """Called when new ticks arrive."""
        for tick in ticks:
            token = tick.get('instrument_token')
            last_price = tick.get('last_price', 0)
            timestamp = tick.get('timestamp') or datetime.now()
            
            # Update price store
            with self._lock:
                self._prices[token] = {
                    'price': last_price,
                    'timestamp': timestamp,
                    'volume': tick.get('volume', 0),
                    'high': tick.get('ohlc', {}).get('high', last_price),
                    'low': tick.get('ohlc', {}).get('low', last_price),
                    'open': tick.get('ohlc', {}).get('open', last_price),
                }
            
            # Fire callback
            if self._on_price_update:
                try:
                    self._on_price_update(token, last_price, timestamp)
                except Exception as e:
                    print(f"   ⚠️ Callback error: {e}")
    
    def _handle_close(self, ws, code, reason):
        """Called when WebSocket closes."""
        self._connected = False
        if self._running:
            print(f"   ⚠️ WebSocket closed: {code} - {reason}")
    
    def _handle_error(self, ws, code, reason):
        """Called on WebSocket error."""
        print(f"   ❌ WebSocket error: {code} - {reason}")
    
    def _handle_reconnect(self, ws, attempts):
        """Called when reconnecting."""
        print(f"   🔄 WebSocket reconnecting... attempt {attempts}")
    
    def get_price(self, token: int) -> Optional[float]:
        """Get latest price for a token (thread-safe)."""
        with self._lock:
            data = self._prices.get(token)
            return data['price'] if data else None
    
    def get_all_prices(self) -> Dict[int, Dict]:
        """Get all cached prices (thread-safe copy)."""
        with self._lock:
            return dict(self._prices)
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected
    
    def stop(self):
        """Stop the ticker gracefully."""
        self._running = False
        if self._ticker:
            try:
                self._ticker.close()
            except:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        print("   🛑 WebSocket ticker stopped.")


class MockTicker:
    """
    Mock ticker for testing without actual WebSocket connection.
    Simulates price updates for paper trading.
    """
    
    def __init__(self):
        self._prices: Dict[int, float] = {}
        self._running = False
    
    def connect(self, tokens: List[int], on_price_update: Optional[Callable] = None):
        """Mock connect - always succeeds."""
        print("   📝 Mock ticker initialized (no real WebSocket)")
        self._running = True
        return True
    
    def set_price(self, token: int, price: float):
        """Manually set a price for testing."""
        self._prices[token] = price
    
    def get_price(self, token: int) -> Optional[float]:
        """Get mocked price."""
        return self._prices.get(token)
    
    def get_all_prices(self) -> Dict[int, float]:
        """Get all mocked prices."""
        return dict(self._prices)
    
    def is_connected(self) -> bool:
        """Mock always returns True when running."""
        return self._running
    
    def stop(self):
        """Stop mock ticker."""
        self._running = False
        print("   📝 Mock ticker stopped.")
