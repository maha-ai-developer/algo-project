import sys
import os
import pandas as pd
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.data.universe_parser import load_nifty_symbols
from infrastructure.llm.client import GeminiAgent
from strategies.fundamental.valuation import DCFModel
from strategies.fundamental.quality import QualityCheck

def run_fundamental_scan():
    print("--- 🧠 STEP 1: FUNDAMENTAL HEALTH CHECK ---")
    
    # 1. Load Universe (Liquidity Filter: >100Cr)
    files = [f for f in os.listdir(config.UNIVERSE_DIR) if f.endswith(".csv")]
    if not files: return
    csv_path = os.path.join(config.UNIVERSE_DIR, files[0])
    
    symbols = load_nifty_symbols(csv_path, min_turnover_cr=0)
    if not symbols: return

    print(f"🔍 Screening {len(symbols)} Liquid Stocks for Financial Health...")
    
    agent = GeminiAgent()
    dcf_engine = DCFModel()
    quality_engine = QualityCheck()
    results = []
    
    for i, sym in enumerate(symbols, 1):
        print(f"\r   👉 [{i}/{len(symbols)}] Checking {sym}...", end="")
        
        data = agent.analyze_fundamentals(sym)
        if not data: continue
            
        fin = data.get('financials', {})
        dcf_in = data.get('dcf_inputs', {})
        qual = data.get('qualitative', {})

        # Quality & Valuation
        check_data = {
            'sales_growth': fin.get('sales_growth_3yr_avg', 0),
            'profit_growth': fin.get('profit_growth_3yr_avg', 0),
            'roe': fin.get('roe_latest', 0),
            'debt_to_equity': fin.get('debt_to_equity', 0)
        }
        q_result = quality_engine.evaluate(check_data)
        
        d_e = check_data['debt_to_equity']
        wacc = dcf_engine.calculate_wacc(
            beta=fin.get('beta', 1.0),
            equity_weight=1/(1+d_e), debt_weight=d_e/(1+d_e),
            cost_of_debt=0.09, tax_rate=dcf_in.get('tax_rate', 0.25)
        )
        
        fcf = dcf_in.get('free_cash_flow_latest_cr', 0)
        g = min(dcf_in.get('growth_rate_projection', 0.10), 0.15)
        projected_fcf = [fcf * ((1 + g) ** n) for n in range(1, 6)]
        
        val_result = dcf_engine.get_intrinsic_value(
            free_cash_flows=projected_fcf, terminal_growth_rate=0.04, wacc=wacc,
            shares_outstanding=dcf_in.get('shares_outstanding_cr', 100),
            net_debt=dcf_in.get('net_debt_cr', 0)
        )

        results.append({
            "Symbol": sym,
            "Quality": q_result['status'],
            "Mgmt Score": qual.get('management_integrity_score', 0),
            "Fair Value": val_result['fair_value'],
            "Buy Price": val_result['buy_price'],
            "Reasoning": qual.get('reasoning', '')[:50]
        })

    print("\n")
    if results:
        df = pd.DataFrame(results)
        # Filter: Only pass INVESTIBLE stocks to next stage
        df_passed = df[df['Quality'] == 'INVESTIBLE'].copy()
        
        df_passed.to_csv(config.FUNDAMENTAL_FILE, index=False)
        print(f"✅ Step 1 Complete. {len(df_passed)} stocks passed to Sector Analysis.")
        print(f"📁 Saved to: {config.FUNDAMENTAL_FILE}")
    else:
        print("❌ No stocks passed quality check.")

if __name__ == "__main__":
    run_fundamental_scan()
