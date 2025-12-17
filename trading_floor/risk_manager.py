import pandas_ta_classic as ta
import pandas as pd
import numpy as np
from datetime import datetime

class RiskManager:
    def __init__(self, total_capital=15000, max_risk_pct=0.02):
        """
        Implements the 'Reduced Total Equity Model'.
        
        Args:
            total_capital (float): Free Cash + Locked-in Profits (Starting Core Equity).
            max_risk_pct (float): Hard stop limit per trade (Default 2%).
        """
        self.core_equity = total_capital  # Updates only on realized PnL
        self.max_risk_pct = max_risk_pct
        
        # Safety: Daily Kill Switch (e.g., 3% of Equity)
        self.max_daily_loss = total_capital * 0.03
        self.current_daily_pnl = 0.0

        print(f"   🛡️ Risk System Online: Capital ₹{self.core_equity:,.2f} | Max Risk {self.max_risk_pct*100}%")

    def update_capital(self, realized_pnl):
        """
        Updates 'Reduced Total Equity' after a trade closes.
        """
        self.core_equity += realized_pnl
        self.current_daily_pnl += realized_pnl
        # print(f"   💰 Capital Updated: ₹{self.core_equity:,.2f} (Daily PnL: ₹{self.current_daily_pnl:,.2f})")

    def _calculate_kelly_factor(self, win_rate, win_loss_ratio):
        """
        Calculates Modified Kelly Criterion.
        Formula: K% = W - ((1 - W) / R)
        """
        if win_loss_ratio <= 0: return 0.0
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Constraints:
        # 1. Negative Kelly = No Edge = No Trade
        # 2. Cap Kelly at 1.0 (100% of Risk Limit) to prevent over-betting
        return max(0.0, min(kelly, 1.0))

    def calculate_atr(self, df, length=14):
        """Calculates ATR (Volatility) from DataFrame."""
        if df is None or len(df) < length: return 0.0
        atr_series = ta.atr(df['high'], df['low'], df['close'], length=length)
        return atr_series.iloc[-1]

    def size_position(self, symbol, entry_price, atr, win_rate=0.5, win_loss_ratio=2.0):
        """
        Determines Position Size using the 'Risk Optimization' Framework.
        
        Steps:
        1. Calculate Max Allowed Risk (Capital * 2%)
        2. Calculate Kelly Factor (Strategy Edge)
        3. Optimize Risk (Max Risk * Kelly)
        4. Calculate Quantity based on ATR Stop Loss
        """
        # 1. Hard Risk Constraint
        max_risk_amt = self.core_equity * self.max_risk_pct
        
        # 2. Strategy Edge (Kelly)
        # If strategy stats are unknown, default to conservative (W=0.4, R=1.5) or provided defaults
        kelly_factor = self._calculate_kelly_factor(win_rate, win_loss_ratio)
        
        # 3. Optimized Risk Amount
        optimized_risk_amt = max_risk_amt * kelly_factor
        
        # Safety: If Kelly is 0 (Bad strategy), don't trade
        if optimized_risk_amt <= 0:
            print(f"   ⚠️ Trade Skipped: Negative/Zero Kelly Factor ({kelly_factor:.2f})")
            return 0, 0, 0

        # 4. Stop Loss Distance (Chandelier Logic: 2x ATR)
        if atr > 0:
            sl_distance = 2.0 * atr
        else:
            # Fallback if ATR fails (0.5% of price)
            sl_distance = entry_price * 0.005
            
        stop_loss_price = entry_price - sl_distance

        # 5. Quantity Calculation
        qty = int(optimized_risk_amt / sl_distance)
        
        # Margin Check (approx 5x leverage for Intraday)
        max_qty_margin = int((self.core_equity * 5) / entry_price)
        final_qty = min(qty, max_qty_margin)

        # Logging the Math (As per your Step 5 request)
        print(f"   📐 Sizing {symbol}: Capital ₹{self.core_equity:.0f} | MaxRisk ₹{max_risk_amt:.0f} | Kelly {kelly_factor:.2f}")
        print(f"   🎯 Optimized Risk: ₹{optimized_risk_amt:.2f} | SL Dist: {sl_distance:.2f} | Qty: {final_qty}")

        return max(1, final_qty), stop_loss_price, sl_distance

    def calculate_chandelier_exit(self, current_price, current_sl, highest_price, atr, direction="LONG"):
        """
        Updates Trailing Stop using ATR Chandelier Logic.
        Long: SL = Highest High - 2x ATR
        Short: SL = Lowest Low + 2x ATR
        """
        multiplier = 2.0
        
        if direction == "LONG":
            # Chandelier Formula
            potential_sl = highest_price - (multiplier * atr)
            
            # Logic: Stop can ONLY move UP
            new_sl = max(current_sl, potential_sl)
            return new_sl
            
        elif direction == "SHORT":
            potential_sl = highest_price + (multiplier * atr) # 'highest_price' acts as Lowest Low here for simplicity in naming
            
            # Logic: Stop can ONLY move DOWN
            new_sl = min(current_sl, potential_sl)
            return new_sl
            
        return current_sl

    def check_kill_switch(self):
        """Returns True if Daily Loss Limit is hit."""
        if self.current_daily_pnl < -self.max_daily_loss:
            print(f"   💀 KILL SWITCH TRIGGERED: Daily Loss ₹{self.current_daily_pnl:.2f}")
            return True
        return False

    def check_time_exit(self):
        """Returns True if it's 3:15 PM IST."""
        now = datetime.now()
        if now.hour == 15 and now.minute >= 15:
            return True
        return False
