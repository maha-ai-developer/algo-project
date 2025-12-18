import pandas as pd
import os

def load_nifty_symbols(csv_path):
    """
    Parses any NSE/Index CSV to extract symbols.
    Index-Agnostic: Works for Nifty 50, Midcap, BankNifty, etc.
    """
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return []

    try:
        # 1. Read first few lines to find the Header Row
        # NSE CSVs often have the Index Name in Row 0 and Headers in Row 1
        header_row = 0
        df_detect = pd.read_csv(csv_path, nrows=5, header=None)
        
        for i, row in df_detect.iterrows():
            row_str = row.astype(str).str.upper().tolist()
            # Check if this row looks like a header (contains "SYMBOL")
            if any("SYMBOL" in col for col in row_str):
                header_row = i
                break
        
        # 2. Read full CSV with correct header
        df = pd.read_csv(csv_path, header=header_row)
        
        # 3. Clean Column Names (Remove spaces/newlines)
        df.columns = df.columns.astype(str).str.strip().str.upper()
        
        # 4. Find the Symbol Column
        symbol_col = None
        candidates = ['SYMBOL', 'TICKER', 'YP_SYMBOL']
        
        for col in df.columns:
            if col in candidates:
                symbol_col = col
                break
        
        if not symbol_col:
            print(f"❌ Could not find 'SYMBOL' column. Found: {df.columns.tolist()}")
            return []

        # 5. Extract and Clean Symbols
        symbols = df[symbol_col].dropna().astype(str).tolist()
        
        # 6. STRICT FILTERING
        clean_symbols = []
        for s in symbols:
            s = s.strip().upper()
            
            # 🛑 CRITICAL FIX: Aggressively filter out Index names
            if "NIFTY" in s: continue    # Skips "NIFTY 50", "NIFTY BANK", etc.
            if s == "SYMBOL": continue   # Skips header repetition
            if s == "": continue         # Skips empty rows
            
            clean_symbols.append(s)
            
        print(f"   ✅ Parsed {len(clean_symbols)} symbols from {os.path.basename(csv_path)}")
        return clean_symbols

    except Exception as e:
        print(f"❌ Error parsing Universe CSV: {e}")
        return []
