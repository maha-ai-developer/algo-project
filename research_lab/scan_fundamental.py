import sys
import os
import pandas as pd
from tabulate import tabulate

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infrastructure.config as config
from infrastructure.data.universe_parser import load_nifty_symbols
from infrastructure.llm.client import GeminiAgent
from strategies.fundamental.valuation import DCFModel
from strategies.fundamental.quality import QualityCheck

def run_fundamental_scan():
    print("--- 🧠 AI FUNDAMENTAL & SECTOR SCANNER ---")
    
    # 1. Load Universe
    if not os.path.exists(config.UNIVERSE_DIR):
        print(f"❌ Directory not found: {config.UNIVERSE_DIR}")
        return

    universe_files = [f for f in os.listdir(config.UNIVERSE_DIR) if f.endswith(".csv")]
    if not universe_files:
        print("❌ No universe CSV found.")
        return
    
    csv_path = os.path.join(config.UNIVERSE_DIR, universe_files[0])
    print(f"📂 Universe: {csv_path}")
    
    symbols = load_nifty_symbols(csv_path)
    if not symbols: return
        
    print(f"🔍 Found {len(symbols)} symbols. Starting Analysis...")
    
    # 2. Initialize Models
    agent = GeminiAgent()
    dcf_engine = DCFModel()
    quality_engine = QualityCheck()
    
    results = []
    
    # 3. Process Each Symbol
    for sym in symbols:
        data = agent.analyze_company(sym)
        
        if not data:
            continue
            
        fin = data.get('financials', {})
        dcf_in = data.get('dcf_inputs', {})
        qual = data.get('qualitative', {})
        sector = data.get('sector', 'OTHERS') 

        # A. Rename keys for Quality Engine
        check_data = {
            'sales_growth': fin.get('sales_growth_3yr_avg', 0),
            'profit_growth': fin.get('profit_growth_3yr_avg', 0),
            'roe': fin.get('roe_latest', 0),
            'debt_to_equity': fin.get('debt_to_equity', 0)
        }
        
        # B. Run Quality Check
        # --- FIXED LINE BELOW ---
        q_result = quality_engine.evaluate(check_data) 
        # Note: q_result is now a dictionary: {'score': 8, 'status': 'INVESTIBLE', ...}
        
        # C. Run Valuation (DCF)
        # Calculate WACC components
        d_e = check_data['debt_to_equity']
        equity_weight = 1 / (1 + d_e)
        debt_weight = d_e / (1 + d_e)
        
        wacc = dcf_engine.calculate_wacc(
            beta=fin.get('beta', 1.0),
            equity_weight=equity_weight,
            debt_weight=debt_weight,
            cost_of_debt=0.09,
            tax_rate=dcf_in.get('tax_rate', 0.25)
        )
        
        # Project Cash Flows (Simplified for scanner: 5 years)
        fcf = dcf_in.get('free_cash_flow_latest_cr', 0)
        g = dcf_in.get('growth_rate_projection', 0.10)
        projected_fcf = [fcf * ((1 + g) ** i) for i in range(1, 6)]
        
        val_result = dcf_engine.get_intrinsic_value(
            free_cash_flows=projected_fcf,
            terminal_growth_rate=0.04,
            wacc=wacc,
            shares_outstanding=dcf_in.get('shares_outstanding_cr', 100),
            net_debt=dcf_in.get('net_debt_cr', 0)
        )

        # D. Store Result
        results.append({
            "Symbol": sym,
            "Sector": sector,
            "Mgmt Score": qual.get('management_integrity_score', 0),
            "Moat": qual.get('moat_rating', 'N/A'),
            "Quality": q_result['status'], # <--- Uses the dict correctly
            "Fair Value": val_result['fair_value'],
            "Buy Price": val_result['buy_price'],
            "AI Reasoning": (qual.get('reasoning') or "")[:50] + "..."
        })

    # 4. Final Report
    if results:
        df = pd.DataFrame(results)
        # Sort by Sector then Quality
        df.sort_values(by=["Sector", "Quality"], ascending=[True, False], inplace=True)
        print("\n" + tabulate(df, headers="keys", tablefmt="grid"))

        save_path = os.path.join(config.ARTIFACTS_DIR, "fundamental_analysis.csv")
        df.to_csv(save_path, index=False)
        print(f"\n✅ Report saved to {save_path}")
    else:
        print("\n❌ No results generated.")

if __name__ == "__main__":
    run_fundamental_scan()
