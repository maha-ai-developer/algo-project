import os
import time
import pandas as pd
from datetime import datetime, timedelta

# NEW IMPORTS
import infrastructure.config as config
from infrastructure.broker.kite_auth import get_kite
from infrastructure.data.instrument_cache import get_instrument_token

# Map user inputs to Kite API constants
INTERVAL_MAP = {
    "1m": "minute", "5m": "5minute", "15m": "15minute", 
    "60m": "60minute", "day": "day"
}

# Max days allowed per request by Kite API
CHUNK_LIMITS = {
    "minute": 60, "5minute": 60, "15minute": 100, 
    "60minute": 360, "day": 2000
}

class DataManager:
    @staticmethod
    def get_csv_path(symbol, timeframe="5m"):
        return os.path.join(config.DATA_DIR, f"{symbol}_{timeframe}.csv")

    @staticmethod
    def load_data(symbol, timeframe="5m"):
        """Loads data from CSV."""
        path = DataManager.get_csv_path(symbol, timeframe)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None

def download_historical_data(symbols, from_date, to_date, interval="5m", output_dir=None):
    """
    Downloads historical data for a list of symbols.
    
    Args:
        symbols: List of stock symbols
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        interval: Data interval ("day", "5m", etc.)
        output_dir: Optional output directory (defaults to DATA_DIR)
    """
    kite = get_kite()
    api_interval = INTERVAL_MAP.get(interval, "5minute")
    
    # Use specified output_dir or default to DATA_DIR
    save_dir = output_dir if output_dir else config.DATA_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    start_dt = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt = datetime.strptime(to_date, "%Y-%m-%d")
    
    print(f"--- 📥 Downloading {interval} data ({from_date} to {to_date}) ---")
    print(f"    📂 Output: {save_dir}")

    for symbol in symbols:
        print(f"🔄 Processing {symbol}...", end=" ")
        
        token = get_instrument_token(symbol)
        if not token:
            print(f"❌ Token not found for {symbol}")
            continue

        # Determine safe chunk size (in days)
        chunk_days = CHUNK_LIMITS.get(api_interval, 60)
        
        all_records = []
        current_start = start_dt

        # --- CHUNKING LOOP ---
        while current_start < end_dt:
            current_end = current_start + timedelta(days=chunk_days)
            if current_end > end_dt:
                current_end = end_dt
            
            try:
                # Fetch Batch
                batch = kite.historical_data(
                    instrument_token=token,
                    from_date=current_start,
                    to_date=current_end,
                    interval=api_interval
                )
                if batch:
                    all_records.extend(batch)
                    
                # Rate limit protection (3 req/sec rule)
                time.sleep(0.4) 
                
            except Exception as e:
                print(f"❌ Error fetching chunk {current_start.date()}: {e}")
                # Don't break, try next chunk or save partial
            
            # Move to next chunk
            current_start = current_end + timedelta(minutes=1)

        if not all_records:
            print(f"⚠️ No data fetched")
            continue

        # Save to specified directory
        df = pd.DataFrame(all_records)
        path = os.path.join(save_dir, f"{symbol}_{interval}.csv")
        df.to_csv(path, index=False)
        print(f"✅ Saved {len(df)} rows")
