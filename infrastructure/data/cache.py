"""
Data Cache Module - Optimizations #1 and #2

#1: Parallel API Calls using ThreadPoolExecutor
#2: Incremental Updates - only fetch delta when cache exists

This eliminates redundant API calls and speeds up data fetching.
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import threading

import infrastructure.config as config


class DataCache:
    """
    Thread-safe data cache with incremental updates and parallel fetching.
    
    Features:
    - In-memory cache with optional disk persistence
    - Incremental updates (only fetch new candles)
    - Parallel bulk fetching with ThreadPoolExecutor
    - Thread-safe access
    """
    
    def __init__(self, kite_client, max_workers: int = 5, lookback_days: int = 120):
        """
        Args:
            kite_client: Authenticated Kite Connect client
            max_workers: Max parallel API requests (Kite allows ~3/sec)
            lookback_days: Default historical data window
        """
        self.kite = kite_client
        self.max_workers = max_workers
        self.lookback_days = lookback_days
        
        # Thread-safe cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._lock = threading.RLock()
        
        # Token map (symbol -> instrument_token)
        self._tokens: Dict[str, int] = {}
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0
    
    def set_tokens(self, token_map: Dict[str, int]):
        """Set instrument token mapping."""
        self._tokens = token_map
    
    def get_data(self, symbol: str, interval: str = "day") -> pd.Series:
        """
        Get price data for a symbol, using cache with incremental updates.
        
        Returns:
            pd.Series: Close prices indexed by date
        """
        cache_key = f"{symbol}_{interval}"
        
        with self._lock:
            if cache_key in self._cache:
                # Cache hit - check if we need incremental update
                df = self._cache[cache_key]
                last_date = df.index[-1] if len(df) > 0 else None
                
                # Only update if last data is older than today
                if last_date and last_date.date() < datetime.now().date():
                    df = self._incremental_update(symbol, interval, df)
                    self._cache[cache_key] = df
                
                self.cache_hits += 1
                return df['close'] if 'close' in df.columns else pd.Series()
            
            # Cache miss - full fetch
            self.cache_misses += 1
            df = self._full_fetch(symbol, interval)
            if df is not None and len(df) > 0:
                self._cache[cache_key] = df
                return df['close']
            return pd.Series()
    
    def _full_fetch(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Full historical data fetch."""
        if symbol not in self._tokens:
            print(f"   ⚠️ Token missing for {symbol}")
            return None
        
        token = self._tokens[symbol]
        to_date = datetime.now()
        from_date = to_date - timedelta(days=self.lookback_days)
        
        try:
            self.api_calls += 1
            data = self.kite.historical_data(token, from_date, to_date, interval)
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            print(f"   ⚠️ API Error for {symbol}: {e}")
            return None
    
    def _incremental_update(self, symbol: str, interval: str, existing_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch only new candles and append to existing data."""
        if symbol not in self._tokens:
            return existing_df
        
        token = self._tokens[symbol]
        last_date = existing_df.index[-1]
        from_date = last_date - timedelta(days=1)  # Small overlap to ensure no gaps
        to_date = datetime.now()
        
        try:
            self.api_calls += 1
            new_data = self.kite.historical_data(token, from_date, to_date, interval)
            if not new_data:
                return existing_df
            
            new_df = pd.DataFrame(new_data)
            new_df['date'] = pd.to_datetime(new_df['date'])
            new_df.set_index('date', inplace=True)
            
            # Combine and remove duplicates
            combined = pd.concat([existing_df, new_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            return combined.sort_index()
            
        except Exception as e:
            print(f"   ⚠️ Incremental update failed for {symbol}: {e}")
            return existing_df
    
    def parallel_fetch(self, symbols: List[str], interval: str = "day") -> Dict[str, pd.Series]:
        """
        Fetch data for multiple symbols in parallel.
        
        Args:
            symbols: List of trading symbols
            interval: Candle interval
            
        Returns:
            Dict mapping symbol -> close price series
        """
        results = {}
        to_fetch = []
        
        # Check cache first
        with self._lock:
            for symbol in symbols:
                cache_key = f"{symbol}_{interval}"
                if cache_key in self._cache:
                    df = self._cache[cache_key]
                    results[symbol] = df['close'] if 'close' in df.columns else pd.Series()
                    self.cache_hits += 1
                else:
                    to_fetch.append(symbol)
        
        if not to_fetch:
            return results
        
        # Parallel fetch for cache misses
        print(f"   ⚡ Parallel fetch: {len(to_fetch)} symbols with {self.max_workers} workers")
        
        def fetch_one(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
            time.sleep(0.35)  # Rate limit: ~3 req/sec
            return sym, self._full_fetch(sym, interval)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fetch_one, sym): sym for sym in to_fetch}
            
            for future in as_completed(futures):
                symbol, df = future.result()
                self.cache_misses += 1
                
                if df is not None and len(df) > 0:
                    with self._lock:
                        self._cache[f"{symbol}_{interval}"] = df
                    results[symbol] = df['close']
                else:
                    results[symbol] = pd.Series()
        
        return results
    
    def get_stats(self) -> Dict:
        """Return cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_calls": self.api_calls,
            "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
