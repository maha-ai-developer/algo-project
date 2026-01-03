"""
AI Advisor - System-Wide Oversight

Provides intelligent analysis of the entire StatArb system:
- Gathers data from all reports (fundamental, sector, pairs, backtest)
- Sends context to Gemini AI for analysis
- Returns actionable recommendations

Usage: python cli.py ai_advisor
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from infrastructure.llm.client import GeminiAgent


def load_json_safe(filepath: str) -> dict:
    """Load JSON file safely, return empty dict on error."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return {}


def load_csv_safe(filepath: str) -> pd.DataFrame:
    """Load CSV file safely, return empty DataFrame on error."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        return pd.read_csv(filepath)
    except Exception:
        return pd.DataFrame()


def gather_system_state() -> dict:
    """
    Gathers the current state of the entire StatArb system.
    
    Returns:
        dict with summarized data from all system components
    """
    context = {}
    
    # 1. PAIRS CANDIDATES
    pairs_file = config.PAIRS_CANDIDATES_FILE
    pairs_data = load_json_safe(pairs_file)
    
    if pairs_data:
        pairs_summary = f"Total pairs: {len(pairs_data)}\n"
        
        # Sample top 5 pairs
        for i, pair in enumerate(pairs_data[:5], 1):
            leg1 = pair.get('leg1', pair.get('stock_y', 'N/A'))
            leg2 = pair.get('leg2', pair.get('stock_x', 'N/A'))
            beta = pair.get('beta', pair.get('hedge_ratio', 0))
            pval = pair.get('p_value', pair.get('adf_pvalue', 0))
            quality = pair.get('quality', 'N/A')
            pairs_summary += f"  {i}. {leg1}-{leg2}: β={beta:.3f}, p={pval:.4f}, quality={quality}\n"
        
        context['pairs_summary'] = pairs_summary
    else:
        context['pairs_summary'] = "No pair candidates found. Run: python cli.py scan_pairs"
    
    # 2. BACKTEST RESULTS
    # 2. BACKTEST RESULTS
    # Prefer full results artifact if available
    backtest_file = os.path.join(config.ARTIFACTS_DIR, "backtest_full_results.json")
    if not os.path.exists(backtest_file):
        backtest_file = os.path.join(config.CACHE_DIR, "backtest_progress.json")
        
    backtest_data = load_json_safe(backtest_file)
    
    # Handle list vs dict format
    results = []
    if isinstance(backtest_data, list):
        results = backtest_data
    elif isinstance(backtest_data, dict) and 'results' in backtest_data:
        results = backtest_data['results']
    
    if results:
        
        # Calculate aggregates
        returns = [r.get('return_pct', 0) for r in results]
        win_rates = [r.get('win_rate', 0) for r in results]
        sharpes = [r.get('sharpe_ratio', 0) for r in results if r.get('sharpe_ratio', 0) > 0]
        
        backtest_summary = f"Total pairs backtested: {len(results)}\n"
        backtest_summary += f"  Avg Return: {sum(returns)/max(len(returns),1):.2f}%\n"
        backtest_summary += f"  Avg Win Rate: {sum(win_rates)/max(len(win_rates),1):.1f}%\n"
        backtest_summary += f"  Avg Sharpe (positive only): {sum(sharpes)/max(len(sharpes),1):.2f}\n"
        
        # Top 3 performers
        top_3 = sorted(results, key=lambda x: x.get('return_pct', 0), reverse=True)[:3]
        backtest_summary += "  Top performers:\n"
        for p in top_3:
            backtest_summary += f"    - {p.get('pair', 'N/A')}: {p.get('return_pct', 0):.2f}% return, Sharpe={p.get('sharpe_ratio', 0):.2f}\n"
        
        context['backtest_summary'] = backtest_summary
    else:
        context['backtest_summary'] = "No backtest results found. Run: python cli.py backtest_pairs"
    
    # 3. FUNDAMENTAL REPORT
    fundamental_file = config.FUNDAMENTAL_FILE
    fundamental_df = load_csv_safe(fundamental_file)
    
    if not fundamental_df.empty:
        fund_summary = f"Stocks analyzed: {len(fundamental_df)}\n"
        
        if 'Score' in fundamental_df.columns:
            avg_score = fundamental_df['Score'].mean()
            fund_summary += f"  Avg Quality Score: {avg_score:.1f}/10\n"
        
        if 'Status' in fundamental_df.columns:
            investible = len(fundamental_df[fundamental_df['Status'] == 'INVESTIBLE'])
            fund_summary += f"  Investible: {investible} | Avoid: {len(fundamental_df) - investible}\n"
        
        context['fundamental_summary'] = fund_summary
    else:
        context['fundamental_summary'] = "No fundamental report found. Run: python cli.py scan_fundamental"
    
    # 4. SECTOR DISTRIBUTION
    sector_file = config.SECTOR_REPORT_FILE
    sector_df = load_csv_safe(sector_file)
    
    if not sector_df.empty and 'Broad_Sector' in sector_df.columns:
        sector_counts = sector_df['Broad_Sector'].value_counts().to_dict()
        sector_summary = f"Total stocks: {len(sector_df)}\n"
        sector_summary += "  Distribution:\n"
        for sector, count in list(sector_counts.items())[:6]:
            sector_summary += f"    - {sector}: {count}\n"
        
        context['sector_summary'] = sector_summary
    else:
        context['sector_summary'] = "No sector report found. Run: python cli.py sector_analysis"
    
    return context


def format_output(result: dict) -> str:
    """Format AI result into a nice display output."""
    
    health = result.get('system_health', 'UNKNOWN')
    health_icons = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    health_icon = health_icons.get(health, "⚪")
    
    output = []
    output.append("")
    output.append("┌" + "─" * 70 + "┐")
    output.append("│" + "🧠 AI ADVISOR REPORT".center(70) + "│")
    output.append("├" + "─" * 70 + "┤")
    
    # Health Status
    health_line = f"{health_icon} SYSTEM HEALTH: {health}"
    output.append("│ " + health_line.ljust(69) + "│")
    reasoning = result.get('health_reasoning', '')[:65]
    output.append("│   " + reasoning.ljust(67) + "│")
    output.append("├" + "─" * 70 + "┤")
    
    # Top Opportunities
    output.append("│ " + "📈 TOP OPPORTUNITIES:".ljust(69) + "│")
    opportunities = result.get('top_opportunities', [])
    if opportunities:
        for i, opp in enumerate(opportunities[:3], 1):
            pair = opp.get('pair', 'N/A')[:20]
            action = opp.get('action', '')[:40]
            priority = opp.get('priority', 'MEDIUM')
            priority_icon = {"HIGH": "🔥", "MEDIUM": "➡️", "LOW": "📌"}.get(priority, "")
            line = f"   {i}. {priority_icon} {pair}: {action}"
            output.append("│ " + line[:69].ljust(69) + "│")
    else:
        output.append("│   " + "No immediate opportunities identified".ljust(67) + "│")
    
    output.append("├" + "─" * 70 + "┤")
    
    # Risk Alerts
    output.append("│ " + "⚠️ RISK ALERTS:".ljust(69) + "│")
    alerts = result.get('risk_alerts', [])
    if alerts:
        for alert in alerts[:3]:
            severity = alert.get('severity', 'INFO')
            severity_icon = {"CRITICAL": "🚨", "WARNING": "⚠️", "INFO": "ℹ️"}.get(severity, "")
            desc = alert.get('description', '')[:60]
            line = f"   {severity_icon} [{severity}] {desc}"
            output.append("│ " + line[:69].ljust(69) + "│")
    else:
        output.append("│   " + "No risk alerts".ljust(67) + "│")
    
    output.append("├" + "─" * 70 + "┤")
    
    # Suggestions
    output.append("│ " + "💡 SUGGESTIONS:".ljust(69) + "│")
    suggestions = result.get('suggestions', [])
    if suggestions:
        for i, sug in enumerate(suggestions[:4], 1):
            line = f"   {i}. {sug[:63]}"
            output.append("│ " + line[:69].ljust(69) + "│")
    else:
        output.append("│   " + "No suggestions".ljust(67) + "│")
    
    # Market Outlook (if available)
    if result.get('market_outlook'):
        output.append("├" + "─" * 70 + "┤")
        output.append("│ " + "📰 MARKET OUTLOOK:".ljust(69) + "│")
        outlook = result.get('market_outlook', '')
        # Wrap long text
        while outlook:
            output.append("│   " + outlook[:65].ljust(67) + "│")
            outlook = outlook[65:]
    
    output.append("└" + "─" * 70 + "┘")
    output.append("")
    
    timestamp = result.get('_timestamp', datetime.now().isoformat())
    output.append(f"  Generated: {timestamp}")
    output.append("")
    
    return "\n".join(output)


def run_ai_advisor():
    """
    Main entry point for AI Advisor CLI command.
    
    Gathers system state, sends to Gemini, and displays recommendations.
    """
    print("\n--- 🧠 AI ADVISOR ---")
    print("   Gathering system state...")
    
    # Step 1: Gather all system data
    context = gather_system_state()
    
    print("   📊 System state loaded:")
    print(f"      - Pairs: {'✅' if 'pairs' not in context.get('pairs_summary', 'No').lower() else '⏳ Pending'}")
    print(f"      - Backtest: {'✅' if 'backtested' in context.get('backtest_summary', 'No').lower() else '⏳ Pending'}")
    print(f"      - Fundamentals: {'✅' if 'analyzed' in context.get('fundamental_summary', 'No').lower() else '⏳ Pending'}")
    print(f"      - Sectors: {'✅' if 'stocks' in context.get('sector_summary', 'No').lower() else '⏳ Pending'}")
    
    print("\n   🤖 Analyzing with Gemini AI...")
    
    # Step 2: Initialize agent and analyze
    try:
        agent = GeminiAgent()
        result = agent.analyze_system_health(context)
    except Exception as e:
        print(f"\n❌ AI analysis failed: {e}")
        print("   Check your GENAI_API_KEY in config.json")
        return
    
    # Step 3: Display results
    output = format_output(result)
    print(output)
    
    # Step 4: Save to file for reference
    report_file = os.path.join(config.ARTIFACTS_DIR, "ai_advisor_report.json")
    try:
        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"   📁 Report saved to: {report_file}")
    except Exception:
        pass


if __name__ == "__main__":
    run_ai_advisor()
