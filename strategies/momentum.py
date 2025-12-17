import pandas_ta_classic as ta
from strategies.base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    def __init__(self, ema_period=50, rsi_period=14, rsi_buy_limit=60, atr_period=14):
        super().__init__("Momentum")
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_buy_limit = rsi_buy_limit
        self.atr_period = atr_period

    def generate_signal(self, df):
        if df is None or len(df) < self.ema_period:
            return {"signal": "NONE", "reason": "Not enough data"}

        calc_df = df.copy()
        calc_df['ema'] = ta.ema(calc_df['close'], length=self.ema_period)
        calc_df['rsi'] = ta.rsi(calc_df['close'], length=self.rsi_period)
        
        # --- NEW: Calculate ATR for Risk Sizing ---
        calc_df['atr'] = ta.atr(calc_df['high'], calc_df['low'], calc_df['close'], length=self.atr_period)

        current = calc_df.iloc[-1]
        price = current['close']
        ema = current['ema']
        rsi = current['rsi']
        atr = current['atr']  # Get ATR Value

        # BUY SIGNAL
        if price > ema and rsi > self.rsi_buy_limit:
            return {
                "signal": "BUY",
                "price": price,
                "atr": atr,  # <--- PASS ATR TO ENGINE
                "reason": f"RSI {rsi:.1f} > {self.rsi_buy_limit}",
                "metadata": {"rsi": rsi, "ema": ema}
            }
        
        # SELL SIGNAL
        elif price < ema:
            return {
                "signal": "SELL",
                "price": price,
                "atr": atr, 
                "reason": f"Price below EMA",
                "metadata": {"rsi": rsi, "ema": ema}
            }

        return {"signal": "NONE", "reason": "No signal"}
