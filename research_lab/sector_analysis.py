"""
Sector Analysis v2.0 - Optimized

Optimizations implemented:
1. Parallel Gemini API calls with ThreadPoolExecutor (3x speedup)
2. JSON caching for API results (instant re-runs)
3. Retry logic with exponential backoff (reliability)
4. Progress persistence for crash recovery
"""

import sys
import os
import json
import time
import pandas as pd
from tabulate import tabulate
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.llm.client import GeminiAgent


# ============================================================
# CONFIGURATION
# ============================================================

MAX_WORKERS = 3          # Parallel API calls (Gemini rate limit friendly)
RETRY_ATTEMPTS = 3       # Retries per symbol on failure
RETRY_DELAY_BASE = 1.5   # Base delay for exponential backoff (seconds)
CACHE_EXPIRY_HOURS = 24  # Cache validity period
API_DELAY = 1.2          # Delay between API calls per worker (rate limiting)


# ============================================================
# CACHE MANAGER (Reused pattern from scan_fundamental)
# ============================================================

class SectorCache:
    """
    JSON-based cache for Gemini sector analysis results.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(config.CACHE_DIR, "sector_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._lock = threading.Lock()
    
    def _get_cache_path(self, symbol: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol}_sector.json")
    
    def get(self, symbol: str) -> Optional[Dict]:
        """Get cached data if valid."""
        path = self._get_cache_path(symbol)
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, 'r') as f:
                cached = json.load(f)
            
            cached_time = datetime.fromisoformat(cached.get('_cached_at', '2000-01-01'))
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            
            if age_hours > CACHE_EXPIRY_HOURS:
                return None
            
            return cached.get('data')
            
        except (json.JSONDecodeError, ValueError):
            return None
    
    def set(self, symbol: str, data: Dict):
        """Cache API result."""
        path = self._get_cache_path(symbol)
        
        with self._lock:
            try:
                with open(path, 'w') as f:
                    json.dump({
                        'data': data,
                        '_cached_at': datetime.now().isoformat(),
                        '_symbol': symbol
                    }, f, indent=2)
            except Exception as e:
                print(f"   ⚠️ Cache write failed for {symbol}: {e}")
    
    def clear(self):
        """Clear all cache."""
        for f in os.listdir(self.cache_dir):
            if f.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, f))


# ============================================================
# PROGRESS MANAGER
# ============================================================

class SectorProgressManager:
    """Saves intermediate progress for crash recovery."""
    
    def __init__(self, progress_file: Optional[str] = None):
        self.progress_file = progress_file or os.path.join(config.CACHE_DIR, "sector_progress.json")
        self._lock = threading.Lock()
        self.results: List[Dict] = []
        self.processed: set = set()
    
    def load(self) -> tuple:
        """Load previous progress if exists."""
        if not os.path.exists(self.progress_file):
            return [], set()
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            processed = set(data.get('processed', []))
            print(f"   📂 Resuming: {len(processed)} symbols already processed")
            return results, processed
        except (json.JSONDecodeError, ValueError):
            return [], set()
    
    def save(self):
        """Save current progress."""
        with self._lock:
            try:
                with open(self.progress_file, 'w') as f:
                    json.dump({
                        'results': self.results,
                        'processed': list(self.processed),
                        '_updated_at': datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                print(f"   ⚠️ Progress save failed: {e}")
    
    def add_result(self, symbol: str, result: Dict):
        """Add result and mark as processed."""
        with self._lock:
            self.results.append(result)
            self.processed.add(symbol)
        
        if len(self.results) % 10 == 0:
            self.save()
    
    def clear(self):
        """Clear progress file."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)


# ============================================================
# API CALLER WITH RETRY
# ============================================================

def fetch_sector_with_retry(
    agent: GeminiAgent,
    symbol: str,
    cache: SectorCache,
    max_retries: int = RETRY_ATTEMPTS
) -> Optional[Dict]:
    """Fetch sector data with caching and exponential backoff retry."""
    
    # Check cache first
    cached = cache.get(symbol)
    if cached:
        return cached
    
    # Rate limiting per call
    time.sleep(API_DELAY)
    
    # Try API with retries
    for attempt in range(max_retries):
        try:
            data = agent.analyze_sector_specifics(symbol)
            
            if data:
                cache.set(symbol, data)
                return data
            
            # Empty response, wait and retry
            time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
            
        except Exception as e:
            delay = RETRY_DELAY_BASE * (2 ** attempt)
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"\n   ❌ {symbol}: Failed after {max_retries} attempts")
    
    return None


# ============================================================
# MAIN ANALYZER (OPTIMIZED)
# ============================================================

def run_sector_analysis(use_cache: bool = True, resume: bool = True, max_workers: int = MAX_WORKERS):
    """
    Optimized sector analysis with parallel processing.
    
    Args:
        use_cache: Use cached API results (default True)
        resume: Resume from previous progress (default True)
        max_workers: Parallel API calls (default 3)
    """
    print("--- 🏭 STEP 2: SECTOR-SPECIFIC ANALYSIS v2.0 (OPTIMIZED) ---")
    print(f"   ⚙️ Workers: {max_workers} | Cache: {use_cache} | Resume: {resume}")
    
    # 1. Load Step 1 Results
    if not os.path.exists(config.FUNDAMENTAL_FILE):
        print("❌ Run scan_fundamental first.")
        return

    df = pd.read_csv(config.FUNDAMENTAL_FILE)
    symbols = df['Symbol'].tolist()
    print(f"📊 Analyzing Sectors for {len(symbols)} Fundamentally Strong Stocks...")
    
    # 2. Initialize components
    agent = GeminiAgent()
    cache = SectorCache() if use_cache else None
    progress = SectorProgressManager()
    
    # 3. Load previous progress if resuming
    if resume:
        progress.results, progress.processed = progress.load()
        symbols = [s for s in symbols if s not in progress.processed]
        print(f"   📊 Remaining: {len(symbols)} symbols to process")
    
    if not symbols and progress.results:
        print("   ✅ All symbols already processed. Using cached results.")
        sector_results = progress.results
    else:
        sector_results = list(progress.results)
        stats = {"success": 0, "failed": 0}
        start_time = time.time()
        
        def process_symbol(sym: str) -> Optional[Dict]:
            """Process a single symbol (runs in thread)."""
            data = fetch_sector_with_retry(agent, sym, cache) if cache else agent.analyze_sector_specifics(sym)
            
            if not data:
                return None
            
            # Extract KPIs safely
            kpis = data.get('sector_kpis', {}) or {}
            
            k1 = kpis.get('kpi_1') or kpis.get('kpi_1_name') or '-'
            k2 = kpis.get('kpi_2') or kpis.get('kpi_2_name') or '-'
            k3 = kpis.get('kpi_3') or kpis.get('kpi_3_name') or '-'
            kpi_str = f"{k1} | {k2} | {k3}"
            
            return {
                "Symbol": sym,
                "Broad_Sector": data.get('broad_sector', 'OTHERS').upper(),
                "Niche_Industry": data.get('niche_industry', '-'),
                "Position": data.get('competitive_position', 'CHALLENGER'),
                "Moat": data.get('moat_rating', 'None'),
                "Key_KPIs": kpi_str
            }
        
        # 4. Parallel processing
        print(f"\n   ⚡ Processing {len(symbols)} symbols with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
            
            for i, future in enumerate(as_completed(futures), 1):
                sym = futures[future]
                
                try:
                    result = future.result()
                    
                    if result:
                        sector_results.append(result)
                        progress.add_result(sym, result)
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                        progress.processed.add(sym)
                    
                    pct = (i / len(symbols)) * 100
                    sys.stdout.write(f"\r   📊 Progress: {i}/{len(symbols)} ({pct:.0f}%) | ✅ {stats['success']} | ❌ {stats['failed']}")
                    sys.stdout.flush()
                    
                except Exception as e:
                    stats["failed"] += 1
                    progress.processed.add(sym)
        
        elapsed = time.time() - start_time
        print(f"\n\n   ⏱️ Completed in {elapsed:.1f}s ({elapsed/max(len(symbols),1):.2f}s/symbol)")
        print(f"   📊 Success: {stats['success']} | Failed: {stats['failed']}")
        
        # Save final progress
        progress.save()
    
    # 5. Generate report
    if sector_results:
        df_sec = pd.DataFrame(sector_results)
        
        if 'Broad_Sector' in df_sec.columns:
            df_sec.sort_values(by=['Broad_Sector', 'Position'], ascending=[True, True], inplace=True)
        
        df_sec.to_csv(config.SECTOR_REPORT_FILE, index=False)
        
        print(f"\n✅ Step 2 Complete. {len(sector_results)} stocks analyzed.")
        print(f"📁 Saved to: {config.SECTOR_REPORT_FILE}")
        
        if len(df_sec) > 0:
            print("\n📋 Sample Results:")
            print(tabulate(df_sec[['Symbol', 'Broad_Sector', 'Position', 'Key_KPIs']].head(10), headers="keys", tablefmt="grid"))
        
        # Clear progress on success
        progress.clear()
    else:
        print("❌ Sector analysis failed. No data was collected.")


def run_sector_analysis_fresh():
    """Run analysis without cache or resume (fresh start)."""
    run_sector_analysis(use_cache=False, resume=False)


if __name__ == "__main__":
    run_sector_analysis()
