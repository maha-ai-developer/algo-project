import pandas as pd
import os

def load_nifty_symbols(csv_path, min_turnover_cr=100):
    """
    Parses NSE CSV and filters by Liquidity (Value Traded).
    
    Args:
        min_turnover_cr (int): Minimum Daily Turnover in Crores (Default: 100Cr)
                               Set to 0 to disable filtering.
    """
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return []

    try:
        # 1. Detect Header
        header_row = 0
        df_detect = pd.read_csv(csv_path, nrows=5, header=None)
        for i, row in df_detect.iterrows():
            row_str = row.astype(str).str.upper().tolist()
            if any("SYMBOL" in col for col in row_str):
                header_row = i
                break
        
        # 2. Read Data
        df = pd.read_csv(csv_path, header=header_row)
        df.columns = df.columns.astype(str).str.strip().str.upper() # Clean Headers
        
        # 3. Identify Columns
        symbol_col = None
        value_col = None
        
        for col in df.columns:
            if col in ['SYMBOL', 'TICKER', 'YP_SYMBOL']:
                symbol_col = col
            if "VALUE" in col or "TURNOVER" in col: # Looks for "VALUE (₹ Crores)"
                value_col = col
                
        if not symbol_col:
            print(f"❌ Could not find SYMBOL column.")
            return []

        # 4. 🧮 THE MATHEMATICAL FILTER (The "Code Trick")
        if value_col and min_turnover_cr > 0:
            # Clean data: Remove commas, handle "-", convert to float
            df[value_col] = df[value_col].astype(str).str.replace(',', '').replace('-', '0')
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
            
            # Filter
            initial_count = len(df)
            df = df[df[value_col] >= min_turnover_cr]
            filtered_count = len(df)
            
            print(f"   💧 Liquidity Filter (>₹{min_turnover_cr}Cr): Reduced {initial_count} -> {filtered_count} stocks.")
        
        # 5. Extract Symbols
        symbols = df[symbol_col].dropna().astype(str).tolist()
        
        # 6. Clean Symbols
        clean_symbols = []
        for s in symbols:
            s = s.strip().upper()
            if "NIFTY" in s: continue
            if s == "SYMBOL": continue
            if s == "": continue
            clean_symbols.append(s)
            
        print(f"   ✅ Loaded {len(clean_symbols)} liquid stocks from {os.path.basename(csv_path)}")
        return clean_symbols

    except Exception as e:
        print(f"❌ Error parsing Universe CSV: {e}")
        return []
