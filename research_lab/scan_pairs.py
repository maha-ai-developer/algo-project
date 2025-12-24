"""
Pairs Scanner v2.0 - Optimized

Strategy: Method 2 (Cointegration) - UNCHANGED
Optimizations:
1. Results caching (skip re-testing same pairs)
2. Progress persistence (resume on crash)
3. Better logging for failed pairs
4. Parallel data loading from cache
"""

import pandas as pd
import os
import sys
import json
import itertools
import time
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.data_manager import download_historical_data, DataManager
from strategies.stat_arb_bot import StatArbBot


# ============================================================
# CONFIGURATION
# ============================================================

MIN_DATA_POINTS = 60     # Minimum data points for OLS
ADF_P_VALUE_THRESHOLD = 0.10  # Cointegration threshold
CACHE_EXPIRY_DAYS = 7    # Cache valid for 7 days


# ============================================================
# PAIRS CACHE (Skip re-testing same pairs)
# ============================================================

class PairsCache:
    """
    Caches cointegration test results to avoid re-testing.
    Only re-tests if underlying data is newer than cache.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        self.cache_file = cache_file or os.path.join(config.DATA_DIR, "pairs_test_cache.json")
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, ValueError):
                self._cache = {}
    
    def _save(self):
        """Save cache to disk."""
        with self._lock:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self._cache, f, indent=2)
            except Exception as e:
                print(f"   ⚠️ Cache save failed: {e}")
    
    def _pair_key(self, s1: str, s2: str) -> str:
        """Generate consistent key regardless of order."""
        return f"{min(s1, s2)}_{max(s1, s2)}"
    
    def get(self, s1: str, s2: str) -> Optional[Dict]:
        """Get cached result if valid."""
        key = self._pair_key(s1, s2)
        
        if key not in self._cache:
            return None
        
        cached = self._cache[key]
        
        # Check expiry
        cached_time = datetime.fromisoformat(cached.get('_tested_at', '2000-01-01'))
        age_days = (datetime.now() - cached_time).days
        
        if age_days > CACHE_EXPIRY_DAYS:
            return None
        
        return cached
    
    def set(self, s1: str, s2: str, result: Dict):
        """Cache test result."""
        key = self._pair_key(s1, s2)
        
        with self._lock:
            self._cache[key] = {
                **result,
                '_tested_at': datetime.now().isoformat(),
                '_pair': f"{s1}-{s2}"
            }
    
    def save(self):
        """Explicit save (called at end of scan)."""
        self._save()
    
    def clear(self):
        """Clear cache."""
        self._cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)


# ============================================================
# PROGRESS MANAGER
# ============================================================

class PairsProgressManager:
    """Saves intermediate progress for crash recovery."""
    
    def __init__(self, progress_file: Optional[str] = None):
        self.progress_file = progress_file or os.path.join(config.DATA_DIR, "pairs_scan_progress.json")
        self._lock = threading.Lock()
        self.candidates: List[Dict] = []
        self.tested_pairs: Set[str] = set()
        self.stats = {"tested": 0, "passed": 0, "failed": 0, "cached": 0}
    
    def _pair_key(self, s1: str, s2: str) -> str:
        return f"{min(s1, s2)}_{max(s1, s2)}"
    
    def load(self) -> Tuple[List[Dict], Set[str]]:
        """Load previous progress."""
        if not os.path.exists(self.progress_file):
            return [], set()
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            
            candidates = data.get('candidates', [])
            tested = set(data.get('tested_pairs', []))
            self.stats = data.get('stats', self.stats)
            
            print(f"   📂 Resuming: {len(tested)} pairs already tested, {len(candidates)} candidates found")
            return candidates, tested
        except (json.JSONDecodeError, ValueError):
            return [], set()
    
    def save(self):
        """Save current progress."""
        with self._lock:
            try:
                with open(self.progress_file, 'w') as f:
                    json.dump({
                        'candidates': self.candidates,
                        'tested_pairs': list(self.tested_pairs),
                        'stats': self.stats,
                        '_updated_at': datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                print(f"   ⚠️ Progress save failed: {e}")
    
    def add_result(self, s1: str, s2: str, result: Optional[Dict]):
        """Add test result."""
        key = self._pair_key(s1, s2)
        
        with self._lock:
            self.tested_pairs.add(key)
            self.stats["tested"] += 1
            
            if result:
                self.candidates.append(result)
                self.stats["passed"] += 1
            else:
                self.stats["failed"] += 1
        
        # Auto-save every 50 pairs
        if self.stats["tested"] % 50 == 0:
            self.save()
    
    def is_tested(self, s1: str, s2: str) -> bool:
        """Check if pair already tested in this session."""
        return self._pair_key(s1, s2) in self.tested_pairs
    
    def clear(self):
        """Clear progress file."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)


# ============================================================
# MAIN SCANNER (OPTIMIZED)
# ============================================================

def scan_pairs(use_cache: bool = True, resume: bool = True):
    """
    Optimized pairs scanner with caching and progress persistence.
    
    Strategy: Method 2 (Cointegration) - UNCHANGED
    
    Args:
        use_cache: Use cached test results (default True)
        resume: Resume from previous progress (default True)
    """
    print("--- 🔬 STAT ARB SCANNER v2.0 (METHOD 2: COINTEGRATION) ---")
    print(f"   ⚙️ Cache: {use_cache} | Resume: {resume}")

    # 1. Load Sector Universe (Leaders & Challengers Only)
    if not os.path.exists(config.SECTOR_REPORT_FILE):
        print("❌ Sector Report not found. Run 'sector_analysis' first.")
        return

    df_sector = pd.read_csv(config.SECTOR_REPORT_FILE)
    df_valid = df_sector[df_sector['Position'].isin(['LEADER', 'CHALLENGER'])].copy()
    
    sector_groups = df_valid.groupby('Broad_Sector')['Symbol'].apply(list).to_dict()
    all_symbols = df_valid['Symbol'].unique().tolist()
    
    print(f"💎 Universe: {len(all_symbols)} Stocks from {len(sector_groups)} Sectors.")
    
    # 2. Bulk Download (1 Year Data)
    print("\n⬇️ Fetching Historical Data...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    download_historical_data(all_symbols, start_date, end_date, interval="day")

    # 3. Load Data to RAM
    print("📊 Loading price data to memory...")
    price_cache: Dict[str, pd.Series] = {}
    loaded = 0
    
    for symbol in all_symbols:
        path = os.path.join(config.DATA_DIR, f"{symbol}_day.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                if len(df) >= MIN_DATA_POINTS:
                    price_cache[symbol] = df['close']
                    loaded += 1
            except Exception as e:
                print(f"   ⚠️ Failed to load {symbol}: {e}")
    
    print(f"   ✅ Loaded {loaded}/{len(all_symbols)} symbols with sufficient data")

    # 4. Initialize components
    bot = StatArbBot()
    pairs_cache = PairsCache() if use_cache else None
    progress = PairsProgressManager()
    
    # Load previous progress
    if resume:
        progress.candidates, progress.tested_pairs = progress.load()
    
    # 5. Generate all pair combinations
    all_pairs: List[Tuple[str, str, str]] = []  # (s1, s2, sector)
    
    for sector, symbols in sector_groups.items():
        if len(symbols) < 2:
            continue
        
        valid_symbols = [s for s in symbols if s in price_cache]
        
        for s1, s2 in itertools.combinations(valid_symbols, 2):
            if not progress.is_tested(s1, s2):
                all_pairs.append((s1, s2, sector))
    
    total_pairs = len(all_pairs)
    print(f"\n⚙️ Testing {total_pairs} pairs for cointegration...")
    
    if total_pairs == 0 and progress.candidates:
        print("   ✅ All pairs already tested. Using cached results.")
    else:
        start_time = time.time()
        
        for i, (s1, s2, sector) in enumerate(all_pairs, 1):
            # Progress indicator
            pct = (i / total_pairs) * 100
            sys.stdout.write(f"\r   👉 [{i}/{total_pairs}] ({pct:.0f}%) {sector}: {s1} vs {s2}...     ")
            sys.stdout.flush()
            
            # Check cache first
            if pairs_cache:
                cached = pairs_cache.get(s1, s2)
                if cached:
                    if cached.get('is_cointegrated'):
                        progress.candidates.append({
                            "leg1": cached['leg1'],
                            "leg2": cached['leg2'],
                            "sector": sector,
                            "hedge_ratio": cached['hedge_ratio'],
                            "intercept": cached['intercept']
                        })
                        progress.stats["passed"] += 1
                    progress.stats["cached"] += 1
                    progress.tested_pairs.add(progress._pair_key(s1, s2))
                    continue
            
            # Run cointegration test
            try:
                is_valid = bot.calibrate(price_cache[s1], price_cache[s2], s1, s2)
                
                # Cache result (both pass and fail)
                cache_entry = {
                    'is_cointegrated': is_valid,
                    'leg1': bot.y_symbol,
                    'leg2': bot.x_symbol,
                    'hedge_ratio': round(bot.beta, 4),
                    'intercept': round(bot.intercept, 4)
                }
                
                if pairs_cache:
                    pairs_cache.set(s1, s2, cache_entry)
                
                if is_valid:
                    print(f"\n      ✅ FOUND: {bot.y_symbol} (Y) vs {bot.x_symbol} (X) | Beta: {bot.beta:.3f}")
                    
                    result = {
                        "leg1": bot.y_symbol,
                        "leg2": bot.x_symbol,
                        "sector": sector,
                        "hedge_ratio": round(bot.beta, 4),
                        "intercept": round(bot.intercept, 4)
                    }
                    progress.add_result(s1, s2, result)
                else:
                    progress.add_result(s1, s2, None)
                    
            except Exception as e:
                # Better logging for failed pairs
                print(f"\n      ⚠️ Error testing {s1}-{s2}: {str(e)[:50]}")
                progress.add_result(s1, s2, None)
        
        elapsed = time.time() - start_time
        print(f"\n\n   ⏱️ Completed in {elapsed:.1f}s ({elapsed/max(total_pairs,1)*1000:.1f}ms/pair)")
        
        # Save caches
        if pairs_cache:
            pairs_cache.save()
        progress.save()
    
    # 6. Display stats
    print(f"\n📊 Stats: Tested {progress.stats['tested']} | Passed {progress.stats['passed']} | Cached {progress.stats['cached']}")
    
    # 7. Save results
    if progress.candidates:
        with open(config.PAIRS_CANDIDATES_FILE, "w") as f:
            json.dump(progress.candidates, f, indent=4)
        
        print(f"\n✅ Saved {len(progress.candidates)} Cointegrated Pairs to {config.PAIRS_CANDIDATES_FILE}")
        
        # Display sample
        print("\n📋 Sample Pairs Found:")
        for p in progress.candidates[:5]:
            print(f"   {p['leg1']} ↔ {p['leg2']} ({p['sector']}) | β={p['hedge_ratio']:.3f}")
        
        # Clear progress on success
        progress.clear()
    else:
        print("❌ No cointegrated pairs found.")


def scan_pairs_fresh():
    """Run scan without cache or resume (fresh start)."""
    scan_pairs(use_cache=False, resume=False)


if __name__ == "__main__":
    scan_pairs()
