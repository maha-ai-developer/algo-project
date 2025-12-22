import pandas as pd
import numpy as np
import os
import sys
import json
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infrastructure.config as config
from strategies.stat_arb_bot import StatArbBot
from strategies.guardian import AssumptionGuardian

class ProfessionalBacktest:
    def __init__(self, capital=100000, transaction_cost_pct=0.001):
        self.capital = capital
        self.cost_pct = transaction_cost_pct
        
    def run(self, pair_data):
        y_sym = pair_data['leg1']
        x_sym = pair_data['leg2']
        initial_beta = pair_data['hedge_ratio']
        intercept = pair_data.get('intercept', 0.0)

        bot = StatArbBot(entry_z=2.0, exit_z=0.0, stop_z=4.0) # Slightly wider stops
        bot.beta = initial_beta
        bot.intercept = intercept
        
        guardian = AssumptionGuardian(lookback_window=60)
        guardian.calibrate(initial_beta)

        try:
            df_y = pd.read_csv(os.path.join(config.DATA_DIR, f"{y_sym}_day.csv"))
            df_x = pd.read_csv(os.path.join(config.DATA_DIR, f"{x_sym}_day.csv"))
            df_y['date'] = pd.to_datetime(df_y['date'])
            df_x['date'] = pd.to_datetime(df_x['date'])
            df_y.set_index('date', inplace=True)
            df_x.set_index('date', inplace=True)
            df = pd.concat([df_y['close'], df_x['close']], axis=1).dropna()
            df.columns = ['Y', 'X']
        except Exception as e:
            return {'error': str(e)}

        position = 0
        pnl = 0.0 # Realized PnL
        equity = self.capital
        trade_log = []
        halt_days = 0
        recalibrations = 0
        
        # Buffers
        hist_y = []
        hist_x = []

        for i in range(len(df)):
            date = df.index[i]
            py = df['Y'].iloc[i]
            px = df['X'].iloc[i]
            
            hist_y.append(py)
            hist_x.append(px)

            # 1. GUARDIAN CHECK
            guardian.update_data(py, px)
            status, reason = guardian.diagnose()
            
            # 2. AUTO-CORRECTION (The Fix for RED status)
            if guardian.needs_recalibration():
                new_beta = guardian.force_recalibrate_to_current()
                bot.beta = new_beta # Update Strategy
                recalibrations += 1
                status = "YELLOW" # We are back in business
            
            if status == "RED":
                halt_days += 1
                if position != 0:
                    # Forced Exit
                    pnl_trade = self._calc_trade_pnl(position, entry_y, entry_x, py, px, bot.beta)
                    equity += pnl_trade
                    trade_log.append({"date": date, "type": "HALT", "pnl": pnl_trade})
                    position = 0
                continue

            # 3. STRATEGY CHECK
            if len(hist_y) < 60: continue
            
            # Rolling Z-Score (No Lookahead)
            # Use last 60 days to calculate mean/std for Z
            window_y = np.array(hist_y[-60:])
            window_x = np.array(hist_x[-60:])
            spread = window_y - (bot.beta * window_x + bot.intercept)
            z_mean = np.mean(spread)
            z_std = np.std(spread) if np.std(spread) > 0 else 1.0
            
            curr_spread = py - (bot.beta * px + bot.intercept)
            z = (curr_spread - z_mean) / z_std
            
            # EXECUTION
            if position == 0:
                if z < -bot.entry_z:
                    position = 1 # Long Spread (Buy Y, Sell X)
                    entry_y, entry_x = py, px
                    equity -= self._cost(py)
                elif z > bot.entry_z:
                    position = -1 # Short Spread (Sell Y, Buy X)
                    entry_y, entry_x = py, px
                    equity -= self._cost(py)
                    
            elif position != 0:
                # Exit Conditions
                take_profit = (position == 1 and z > -bot.exit_z) or (position == -1 and z < bot.exit_z)
                stop_loss = (abs(z) > bot.stop_z)
                
                if take_profit or stop_loss:
                    pnl_trade = self._calc_trade_pnl(position, entry_y, entry_x, py, px, bot.beta)
                    equity += pnl_trade
                    trade_log.append({"date": date, "type": "EXIT", "pnl": pnl_trade})
                    position = 0

        # Stats
        total_ret = ((equity - self.capital) / self.capital) * 100
        return {
            "pair": f"{y_sym}-{x_sym}",
            "leg1": y_sym, "leg2": x_sym,
            "return_pct": round(total_ret, 2),
            "trades": len(trade_log),
            "halt_days": halt_days,
            "recalibrations": recalibrations,
            "final_beta": round(bot.beta, 4),
            "final_intercept": round(bot.intercept, 4)
        }

    def _calc_trade_pnl(self, pos, ey, ex, cy, cx, beta):
        # Dollar Neutral PnL Approximation
        # Size = 50k per leg
        qty_y = 50000 / ey
        qty_x = (50000 * beta) / ex # Beta weighted
        
        if pos == 1: # Long Y, Short X
            return (cy - ey)*qty_y + (ex - cx)*qty_x - self._cost(cy)
        else:        # Short Y, Long X
            return (ey - cy)*qty_y + (cx - ex)*qty_x - self._cost(cy)

    def _cost(self, price):
        return (50000 * self.cost_pct)

def run_pro_backtest():
    print("--- 🧪 PROFESSIONAL BACKTEST (ADAPTIVE AGENT) ---")
    
    if not os.path.exists(config.PAIRS_CANDIDATES_FILE):
        print("❌ No candidates found.")
        return
        
    with open(config.PAIRS_CANDIDATES_FILE, "r") as f:
        candidates = json.load(f)
        
    engine = ProfessionalBacktest()
    results = []
    
    # Run Simulation
    for pair in candidates:
        print(f"Testing {pair['leg1']}-{pair['leg2']}...", end="\r")
        res = engine.run(pair)
        if 'error' not in res:
            results.append(res)
            
    if not results:
        print("❌ No valid results.")
        return

    # Filter Winners
    # Criteria: Positive Return AND NOT broken (Halt days < 150)
    df = pd.DataFrame(results)
    winners = df[ (df['return_pct'] > 2.0) & (df['halt_days'] < 150) ].copy()
    winners = winners.sort_values(by='return_pct', ascending=False)

    print("\n\n🏆 TOP PERFORMING PAIRS (LIVE READY)")
    print(tabulate(winners[['pair', 'return_pct', 'trades', 'halt_days', 'recalibrations']].head(15), 
                   headers="keys", tablefmt="simple_grid"))

    # --- SAVE TO FILE (The Fix) ---
    if not winners.empty:
        live_config = []
        for _, row in winners.head(10).iterrows(): # Take Top 10
            live_config.append({
                "leg1": row['leg1'],
                "leg2": row['leg2'],
                "hedge_ratio": row['final_beta'], # Use the adapted beta
                "intercept": row['final_intercept'],
                "strategy": "StatArb_Method2_Adaptive"
            })
            
        with open(config.PAIRS_CONFIG, "w") as f:
            json.dump(live_config, f, indent=4)
        print(f"\n✅ Saved {len(live_config)} pairs to {config.PAIRS_CONFIG}")
        print("🚀 You can now run 'python cli.py engine --mode PAPER'")
    else:
        print("\n❌ No pairs met the profit criteria (Return > 2%).")

if __name__ == "__main__":
    run_pro_backtest()
