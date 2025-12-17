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
    print("--- 🧠 AI FUNDAMENTAL SCANNER ---")
    
    # 1. Load Universe
    if not os.path.exists(config.UNIVERSE_DIR):
        print(f"❌ Directory not found: {config.UNIVERSE_DIR}")
        return

    universe_files = [f for f in os.listdir(config.UNIVERSE_DIR) if f.endswith(".csv")]
    if not universe_files:
        print("❌ No universe CSV found in data/universe/")
        print("   -> Please download the NIFTY 50 CSV from NSE and place it there.")
        return
    
    csv_path = os.path.join(config.UNIVERSE_DIR, universe_files[0]) # Pick the first one
    print(f"📂 Universe: {csv_path}")
    
    symbols = load_nifty_symbols(csv_path)
    
    # --- FIX: Stop here if no symbols found ---
    if not symbols:
        print("❌ Error: Found 0 symbols. Check the CSV format.")
        return
        
    print(f"🔍 Found {len(symbols)} symbols. Starting Deep Research...\n")

    # 2. Initialize Agents
    try:
        agent = GeminiAgent()
    except Exception as e:
        print(f"❌ AI Agent Init Failed: {e}")
        print("   -> Check your config.json for 'genai' API key.")
        return

    dcf_engine = DCFModel()
    quality_engine = QualityCheck()
    
    results = []

    # 3. Analyze Loop (Running on first 5 for testing)
    # Remove [:5] to run on all stocks
    for i, sym in enumerate(symbols): 
        print(f"[{i+1}/{len(symbols)}] 🔬 Analyzing {sym}...")
        
        # A. AI Extraction
        data = agent.analyze_company(sym)
        if not data: 
            print(f"   ⚠️ Skipping {sym} (AI Extraction Failed)")
            continue

        fin = data.get('financials', {})
        dcf_in = data.get('dcf_inputs', {})
        qual = data.get('qualitative', {})

        # B. Quality Check
        quality_input = {
            "sales_growth": fin.get('sales_growth_3yr_avg', 0),
            "profit_growth": fin.get('profit_growth_3yr_avg', 0),
            "roe": fin.get('roe_latest', 0),
            "debt_to_equity": fin.get('debt_to_equity', 0)
        }
        q_result = quality_engine.evaluate(quality_input)
        
        # C. Valuation Check (DCF)
        base_fcf = dcf_in.get('free_cash_flow_latest_cr', 0)
        growth = dcf_in.get('growth_rate_projection', 0.10)
        
        # Simple projection
        projected_fcf = [base_fcf * ((1 + growth) ** y) for y in range(1, 6)]
        
        # Estimate WACC inputs if missing
        d_e = fin.get('debt_to_equity', 0.5)
        equity_weight = 1 / (1 + d_e)
        debt_weight = d_e / (1 + d_e)
        
        wacc = dcf_engine.calculate_wacc(
            beta=fin.get('beta', 1.0),
            equity_weight=equity_weight,
            debt_weight=debt_weight,
            cost_of_debt=0.09,
            tax_rate=dcf_in.get('tax_rate', 0.25)
        )
        
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
            "Mgmt Score": qual.get('management_integrity_score', 0),
            "Moat": qual.get('moat_rating', 'N/A'),
            "Quality": q_result['status'],
            "Fair Value": val_result['fair_value'],
            "Buy Price": val_result['buy_price'],
            "AI Reasoning": (qual.get('reasoning') or "")[:50] + "..."
        })

    # 4. Final Report
    if results:
        df = pd.DataFrame(results)
        print("\n" + tabulate(df, headers="keys", tablefmt="grid"))
        
        save_path = os.path.join(config.ARTIFACTS_DIR, "fundamental_analysis.csv")
        df.to_csv(save_path, index=False)
        print(f"\n✅ Report saved to {save_path}")
    else:
        print("\n❌ No results generated.")

if __name__ == "__main__":
    run_fundamental_scan()
