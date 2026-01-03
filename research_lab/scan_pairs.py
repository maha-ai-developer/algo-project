"""
Pairs Scanner v3.0 - Enhanced with Core Module

Strategy: Method 2 (Cointegration) with:
1. Error ratio optimization (optimal X/Y selection)
2. Intercept risk validation (reject >70%)
3. Quality scoring (EXCELLENT/GOOD/FAIR/POOR)

Uses new core/ module for consistent pair analysis.
"""

import pandas as pd
import os
import sys
import json
import itertools
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.data_manager import download_historical_data

# Import new core module
from core import (
    analyze_pair_from_prices,
    calculate_optimal_direction_from_prices,
    assess_intercept_risk,
    LOOKBACK_PERIOD,
    ADF_THRESHOLD,
    QUALITY_EXCELLENT,
    QUALITY_GOOD,
    QUALITY_FAIR,
    calculate_hurst_exponent
)
from core.constants import INTERCEPT_HIGH_RISK, HURST_THRESHOLD, ADF_THRESHOLD


# ============================================================
# CONFIGURATION
# ============================================================

MIN_DATA_POINTS = 60         # Minimum data points for analysis
ADF_P_VALUE_THRESHOLD = ADF_THRESHOLD # Strict 0.01 from core
CACHE_EXPIRY_DAYS = 7        # Cache valid for 7 days
MAX_INTERCEPT_PERCENT = 70   # Reject if intercept > 70% of Y price
MIN_R_SQUARED = 0.64         # CRITICAL: R² > 0.64 = Correlation > 0.8
MAX_HALF_LIFE_DAYS = 30      # Reject if mean-reversion takes > 30 days


# ============================================================
# PAIRS CACHE (Skip re-testing same pairs)
# ============================================================

class PairsCache:
    """
    Caches cointegration test results to avoid re-testing.
    Only re-tests if underlying data is newer than cache.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        self.cache_file = cache_file or os.path.join(config.CACHE_DIR, "pairs_test_cache.json")
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
        self.progress_file = progress_file or os.path.join(config.CACHE_DIR, "pairs_scan_progress.json")
        self._lock = threading.Lock()
        self.candidates: List[Dict] = []
        self.tested_pairs: Set[str] = set()
        self.stats = {"tested": 0, "passed": 0, "failed": 0, "cached": 0, "rejected_intercept": 0}
    
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
    
    def add_result(self, s1: str, s2: str, result: Optional[Dict], rejected_reason: str = None):
        """Add test result."""
        key = self._pair_key(s1, s2)
        
        with self._lock:
            self.tested_pairs.add(key)
            self.stats["tested"] += 1
            
            if result:
                self.candidates.append(result)
                self.stats["passed"] += 1
            elif rejected_reason == "intercept":
                self.stats["rejected_intercept"] += 1
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
# MAIN SCANNER (v3.0 with Core Module)
# ============================================================

def scan_pairs(use_cache: bool = True, resume: bool = True):
    """
    Enhanced pairs scanner with core module integration.
    
    NEW in v3.0:
    - Error ratio optimization (optimal X/Y selection)
    - Intercept risk validation (reject if >70%)
    - Quality scoring from core module
    
    Args:
        use_cache: Use cached test results (default True)
        resume: Resume from previous progress (default True)
    """
    print("--- 🔬 STAT ARB SCANNER v3.0 (WITH CORE MODULE) ---")
    print(f"   ⚙️ Cache: {use_cache} | Resume: {resume}")
    print(f"   📊 Using core module for pair analysis")

    # 1. Load Sector Universe (Leaders & Challengers Only)
    if not os.path.exists(config.SECTOR_REPORT_FILE):
        print("❌ Sector Report not found. Run 'sector_analysis' first.")
        return

    df_sector = pd.read_csv(config.SECTOR_REPORT_FILE)
    df_valid = df_sector[df_sector['Position'].isin(['LEADER', 'CHALLENGER'])].copy()
    
    sector_groups = df_valid.groupby('Broad_Sector')['Symbol'].apply(list).to_dict()
    all_symbols = df_valid['Symbol'].unique().tolist()
    
    print(f"💎 Universe: {len(all_symbols)} Stocks from {len(sector_groups)} Sectors.")
    
    # 2. Bulk Download to PAIR_SELECTION folder (250 days per research recommendation)
    print("\n⬇️ Fetching Historical Data for Pair Selection...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=config.PAIR_SELECTION_DAYS)).strftime("%Y-%m-%d")
    download_historical_data(all_symbols, start_date, end_date, interval="day",
                             output_dir=config.PAIR_SELECTION_DIR)

    # 3. Load Data to RAM
    print("📊 Loading price data to memory...")
    price_cache: Dict[str, pd.Series] = {}
    loaded = 0
    
    for symbol in all_symbols:
        path = os.path.join(config.PAIR_SELECTION_DIR, f"{symbol}_day.csv")
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
                if cached and cached.get('is_valid'):
                    progress.candidates.append({
                        "leg1": cached['y_stock'],
                        "leg2": cached['x_stock'],
                        "sector": sector,
                        "intercept": cached.get('intercept', 0),
                        "beta": cached.get('beta', 0),
                        "adf_pvalue": cached.get('adf_pvalue', 0),
                        "sigma": cached.get('residual_std_dev', 0),
                        "quality": cached.get('quality', 'UNKNOWN'),
                        "intercept_percent": cached.get('intercept_percent', 0),
                        "error_ratio": cached.get('error_ratio', 0)
                    })
                    progress.stats["passed"] += 1
                    progress.stats["cached"] += 1
                    progress.tested_pairs.add(progress._pair_key(s1, s2))
                    continue
                elif cached:  # Cache exists but not valid
                    progress.stats["cached"] += 1
                    progress.tested_pairs.add(progress._pair_key(s1, s2))
                    continue
            
            # NEW: Use core module for pair analysis
            try:
                prices_1 = price_cache[s1].values
                prices_2 = price_cache[s2].values
                
                # Step 1: Determine optimal X/Y using error ratio
                optimal = calculate_optimal_direction_from_prices(prices_1, prices_2, s1, s2)
                
                # Get optimal X and Y (returns 'X' and 'Y' keys)
                sym_y = optimal['Y']
                sym_x = optimal['X']
                prices_y = optimal['Y_prices']
                prices_x = optimal['X_prices']
                
                # Step 2: Full pair analysis with core module
                # Note: analyze_pair_from_prices determines optimal X/Y internally
                # We pass prices in order and let it optimize
                pair = analyze_pair_from_prices(
                    prices_a=prices_x,
                    prices_b=prices_y,
                    symbol_a=sym_x,
                    symbol_b=sym_y,
                    sector=sector
                )
                
                # Step 3: Check ADF (stationarity)
                if not pair.is_stationary:
                    cache_entry = {
                        'is_valid': False,
                        'reason': 'not_stationary',
                        'adf_pvalue': pair.adf_value
                    }
                    if pairs_cache:
                        pairs_cache.set(s1, s2, cache_entry)
                    progress.add_result(s1, s2, None)
                    continue
                
                # Step 4: NEW - Check intercept risk
                current_price_y = float(prices_y[-1])
                current_price_x = float(prices_x[-1])
                
                intercept_percent = abs(pair.intercept / current_price_y * 100) if current_price_y > 0 else 100
                
                if intercept_percent > MAX_INTERCEPT_PERCENT:
                    print(f"\n      🚫 REJECTED: {sym_y}/{sym_x} | Intercept {intercept_percent:.0f}% > {MAX_INTERCEPT_PERCENT}%")
                    cache_entry = {
                        'is_valid': False,
                        'reason': 'high_intercept',
                        'intercept_percent': intercept_percent
                    }
                    if pairs_cache:
                        pairs_cache.set(s1, s2, cache_entry)
                    progress.add_result(s1, s2, None, rejected_reason="intercept")
                    continue
                
                # Step 5: NEW - Check R² (critical per AI analysis)
                # Pairs with R² < 0.40 are NOT truly cointegrated and will blow up
                from scipy import stats
                _, _, r_value, _, _ = stats.linregress(prices_x, prices_y)
                r_squared = r_value ** 2
                
                if r_squared < MIN_R_SQUARED:
                    print(f"\n      🚫 REJECTED: {sym_y}/{sym_x} | R²={r_squared:.3f} < {MIN_R_SQUARED}")
                    cache_entry = {
                        'is_valid': False,
                        'reason': 'low_r_squared',
                        'r_squared': r_squared
                    }
                    if pairs_cache:
                        pairs_cache.set(s1, s2, cache_entry)
                    progress.add_result(s1, s2, None, rejected_reason="r_squared")
                    continue
                
                # Step 6: NEW - Calculate Half-Life of Mean Reversion
                # Half-Life = -log(2) / log(1 + theta) where theta is from AR(1) regression
                # If spread takes too long to revert, it's not a good pair
                import numpy as np
                residuals = pair.residuals
                if len(residuals) > 10:
                    # AR(1) regression: residual_t = theta * residual_{t-1} + epsilon
                    y = residuals[1:]
                    x = residuals[:-1]
                    theta = np.corrcoef(x, y)[0, 1]  # Autocorrelation
                    
                    if theta > 0 and theta < 1:
                        half_life = -np.log(2) / np.log(theta)
                    else:
                        half_life = 999  # Non-mean-reverting
                else:
                    half_life = 999
                
                if half_life > MAX_HALF_LIFE_DAYS:
                    print(f"\n      🚫 REJECTED: {sym_y}/{sym_x} | Half-Life={half_life:.1f} days > {MAX_HALF_LIFE_DAYS}")
                    cache_entry = {
                        'is_valid': False,
                        'reason': 'slow_mean_reversion',
                        'half_life': half_life
                    }
                    if pairs_cache:
                        pairs_cache.set(s1, s2, cache_entry)
                    progress.add_result(s1, s2, None, rejected_reason="half_life")
                    continue
                
                # Step 7: NEW - Hurst Exponent (Strict Mean Reversion)
                hurst = calculate_hurst_exponent(pair.residuals)
                if hurst > HURST_THRESHOLD:
                    print(f"\n      🚫 REJECTED: {sym_y}/{sym_x} | Hurst={hurst:.2f} > {HURST_THRESHOLD} (Trending)")
                    cache_entry = {
                        'is_valid': False,
                        'reason': 'hurst_trending',
                        'hurst': hurst
                    }
                    if pairs_cache:
                        pairs_cache.set(s1, s2, cache_entry)
                    progress.add_result(s1, s2, None, rejected_reason="hurst")
                    continue

                
                # Step 7: PASSED ALL CHECKS - Save validated pair
                explained_percent = 100 - intercept_percent
                
                print(f"\n      ✅ FOUND: {sym_y} (Y) vs {sym_x} (X)")
                print(f"         β={pair.beta:.3f} | ADF={pair.adf_value:.4f} (p<{ADF_THRESHOLD})")
                print(f"         R²={r_squared:.3f} | Half-Life={half_life:.1f}d | Hurst={hurst:.2f}")
                print(f"         Quality: {pair.quality} | Z-Score: {pair.z_score:.2f}")
                
                result = {
                    "leg1": sym_y,  # Y (Dependent)
                    "leg2": sym_x,  # X (Independent)
                    "stock_y": sym_y,
                    "stock_x": sym_x,
                    "sector": sector,
                    "intercept": round(pair.intercept, 4),
                    "beta": round(pair.beta, 4),
                    "adf_pvalue": round(pair.adf_value, 4),
                    "sigma": round(pair.residual_std_dev, 4),
                    "r_squared": round(r_squared, 4),
                    "quality": pair.quality,
                    "intercept_percent": round(intercept_percent, 1),
                    "explained_percent": round(explained_percent, 1),
                    "error_ratio": round(pair.error_ratio, 4),
                    "z_score": round(pair.z_score, 2)
                }
                
                # Cache validated result
                cache_entry = {
                    'is_valid': True,
                    'y_stock': sym_y,
                    'x_stock': sym_x,
                    **result
                }
                if pairs_cache:
                    pairs_cache.set(s1, s2, cache_entry)
                
                progress.add_result(s1, s2, result)
                    
            except Exception as e:
                print(f"\n      ⚠️ Error testing {s1}-{s2}: {str(e)[:50]}")
                progress.add_result(s1, s2, None)
        
        elapsed = time.time() - start_time
        print(f"\n\n   ⏱️ Completed in {elapsed:.1f}s ({elapsed/max(total_pairs,1)*1000:.1f}ms/pair)")
        
        # Save caches
        if pairs_cache:
            pairs_cache.save()
        progress.save()
    
    # 6. Display stats
    print(f"\n📊 Stats:")
    print(f"   Tested: {progress.stats['tested']}")
    print(f"   Passed: {progress.stats['passed']}")
    print(f"   Rejected (high intercept): {progress.stats['rejected_intercept']}")
    print(f"   Cached: {progress.stats['cached']}")
    
    # 7. Save results
    if progress.candidates:
        # Sort by quality (EXCELLENT > GOOD > FAIR > POOR)
        quality_order = {QUALITY_EXCELLENT: 0, QUALITY_GOOD: 1, QUALITY_FAIR: 2, 'POOR': 3}
        progress.candidates.sort(key=lambda x: quality_order.get(x.get('quality', 'POOR'), 3))
        
        with open(config.PAIRS_CANDIDATES_FILE, "w") as f:
            json.dump(progress.candidates, f, indent=4)
        
        print(f"\n✅ Saved {len(progress.candidates)} VALIDATED Pairs to {config.PAIRS_CANDIDATES_FILE}")
        
        # Create pair_data.csv with enhanced columns
        csv_path = os.path.join(config.ARTIFACTS_DIR, "pair_data.csv")
        with open(csv_path, "w") as f:
            # Enhanced header
            f.write("sector,yStock,xStock,intercept,beta,adf_pvalue,sigma,quality,intercept_pct,explained_pct,z_score\n")
            for p in progress.candidates:
                f.write(f"{p.get('sector','')},{p.get('leg1','')},{p.get('leg2','')},")
                f.write(f"{p.get('intercept',0)},{p.get('beta',0)},{p.get('adf_pvalue',0)},")
                f.write(f"{p.get('sigma',0)},{p.get('quality','')},{p.get('intercept_percent',0)},")
                f.write(f"{p.get('explained_percent',0)},{p.get('z_score',0)}\n")
        
        print(f"📄 Saved pair_data.csv to {csv_path}")
        
        # Display top pairs by quality
        print("\n📋 TOP VALIDATED PAIRS:")
        for p in progress.candidates[:10]:
            quality_icon = "🌟" if p.get('quality') == QUALITY_EXCELLENT else "✅" if p.get('quality') == QUALITY_GOOD else "⚠️"
            print(f"   {quality_icon} {p['leg1']}/{p['leg2']} ({p['sector']})")
            print(f"      β={p['beta']:.3f} | ADF={p['adf_pvalue']:.4f} | Explains {p.get('explained_percent',0):.0f}%")
        
        # Clear progress on success
        progress.clear()
    else:
        print("❌ No validated cointegrated pairs found.")


def scan_pairs_fresh():
    """Run scan without cache or resume (fresh start)."""
    scan_pairs(use_cache=False, resume=False)


if __name__ == "__main__":
    scan_pairs()
