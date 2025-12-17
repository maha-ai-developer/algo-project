import pandas as pd
import os

class UniverseBuilder:
    def __init__(self):
        pass

    def load_csv_universe(self, csv_path):
        """
        Reads NIFTY 500 CSV and returns a clean list of symbols.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        
        # 1. Identify the Symbol Column
        possible_cols = [c for c in df.columns if "SYMBOL" in c.upper() or "TICKER" in c.upper()]
        
        if not possible_cols:
            raise Exception(f"Could not find a 'Symbol' column in CSV. Found: {df.columns.tolist()}")
            
        target_col = possible_cols[0]
        
        # 2. Extract and Clean
        raw_symbols = df[target_col].dropna().astype(str).tolist()
        
        symbols = []
        for s in raw_symbols:
            clean_s = s.strip().upper()
            
            # FILTER: Remove Indices and Header Junk
            if "NIFTY" in clean_s or "SYMBOL" in clean_s:
                continue
                
            symbols.append(clean_s)
        
        return symbols

    @staticmethod
    def save_symbols(symbols, filepath="symbols.txt"):
        with open(filepath, "w") as f:
            for s in symbols:
                f.write(f"{s}\n")
    
    # Keeping existing methods for backward compatibility
    def top_by_price(self, n=50):
        # Placeholder for your existing logic if needed
        return []
        
    def top_by_volume(self, n=50):
        # Placeholder for your existing logic if needed
        return []
