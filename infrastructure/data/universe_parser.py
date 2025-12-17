import pandas as pd
import os

def load_nifty_symbols(csv_path):
    """
    Parses the NSE universe CSV.
    Handles tricky headers like "SYMBOL \n" and filters out "NIFTY 50".
    """
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return []

    try:
        # 1. Read CSV (Header is usually on Row 0)
        df = pd.read_csv(csv_path)
        
        # 2. Clean Column Names (Remove newlines, spaces, quotes)
        df.columns = df.columns.astype(str).str.replace('\n', '').str.strip()
        
        # Debug: Print columns to see what pandas found
        # print(f"DEBUG: Parsed Columns: {df.columns.tolist()}")

        # 3. Find the Symbol Column
        symbol_col = None
        candidates = ['SYMBOL', 'TICKER', 'YP_SYMBOL']
        
        for col in df.columns:
            if col.upper() in candidates:
                symbol_col = col
                break
        
        if not symbol_col:
            # Fallback search
            for col in df.columns:
                if 'SYMBOL' in col.upper():
                    symbol_col = col
                    break
        
        if symbol_col:
            # 4. Extract Symbols
            symbols = df[symbol_col].dropna().astype(str).tolist()
            
            # 5. Filter Garbage (Remove "NIFTY 50" and empty strings)
            clean_symbols = [
                s.strip() for s in symbols 
                if s.strip().upper() != "NIFTY 50" and s.strip() != ""
            ]
            
            return clean_symbols
        else:
            print(f"❌ Could not find 'SYMBOL' column. Columns found: {df.columns.tolist()}")
            return []
            
    except Exception as e:
        print(f"❌ Error parsing universe: {e}")
        return []
