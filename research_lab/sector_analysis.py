import sys
import os
import time  # <--- NEW: Needed for pacing
import pandas as pd
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.llm.client import GeminiAgent

def run_sector_analysis():
    print("--- 🏭 STEP 2: SECTOR-SPECIFIC ANALYSIS (VARSITY LOGIC) ---")
    
    # 1. Load Step 1 Results
    if not os.path.exists(config.FUNDAMENTAL_FILE):
        print("❌ Run scan_fundamental first.")
        return

    df = pd.read_csv(config.FUNDAMENTAL_FILE)
    symbols = df['Symbol'].tolist()
    print(f"📊 Analyzing Sectors for {len(symbols)} Fundamentally Strong Stocks...")
    
    agent = GeminiAgent()
    sector_results = []
    
    # Counter for stats
    success_count = 0
    fail_count = 0
    
    for i, sym in enumerate(symbols, 1):
        # Print current status (overwrite line)
        sys.stdout.write(f"\r   👉 [{i}/{len(symbols)}] Deep Dive: {sym}...")
        sys.stdout.flush()
        
        # --- 🛡️ RATE LIMIT PROTECTION ---
        # Google limits free/standard tiers. We must pause.
        time.sleep(4) 
        
        try:
            data = agent.analyze_sector_specifics(sym)
            
            if not data:
                # print(f"\n      ⚠️ {sym}: No Data (AI Error or Rate Limit)")
                fail_count += 1
                continue
            
            # Extract KPIs safely
            kpis = data.get('sector_kpis', {})
            # Handle if kpis is None or empty
            if not kpis: kpis = {}
                
            # Create a string representation
            # We check multiple possible key formats the AI might return
            k1 = kpis.get('kpi_1') or kpis.get('kpi_1_name') or '-'
            k2 = kpis.get('kpi_2') or kpis.get('kpi_2_name') or '-'
            k3 = kpis.get('kpi_3') or kpis.get('kpi_3_name') or '-'
            
            kpi_str = f"{k1} | {k2} | {k3}"

            sector_results.append({
                "Symbol": sym,
                "Broad_Sector": data.get('broad_sector', 'OTHERS').upper(),
                "Niche_Industry": data.get('niche_industry', '-'),
                "Position": data.get('competitive_position', 'CHALLENGER'),
                "Moat": data.get('moat_rating', 'None'),
                "Key_KPIs": kpi_str
            })
            success_count += 1
            
        except Exception as e:
            print(f"\n      ❌ Error processing {sym}: {e}")
            fail_count += 1
            continue

    print("\n") # Clean finish
    print(f"📊 Stats: {success_count} Succeeded | {fail_count} Failed")

    if sector_results:
        df_sec = pd.DataFrame(sector_results)
        
        # Sort by Sector to see peers together
        if 'Broad_Sector' in df_sec.columns:
            df_sec.sort_values(by=['Broad_Sector', 'Position'], ascending=[True, True], inplace=True)
        
        df_sec.to_csv(config.SECTOR_REPORT_FILE, index=False)
        
        print(tabulate(df_sec[['Symbol', 'Broad_Sector', 'Position', 'Key_KPIs']].head(10), headers="keys", tablefmt="grid"))
        print(f"✅ Step 2 Complete. Report saved to {config.SECTOR_REPORT_FILE}")
    else:
        print("❌ Sector analysis failed. No data was collected.")
        print("   -> Check your Internet connection or API Key Quota.")

if __name__ == "__main__":
    run_sector_analysis()
