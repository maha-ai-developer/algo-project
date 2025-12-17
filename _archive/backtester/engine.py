import pandas as pd
import pandas_ta_classic as ta
from strategies.momentum_strategy import MomentumStrategy

def run_backtest(csv_path, symbol, exchange="NSE", timeframe="5m", capital=100000):
    """
    Simulates a backtest using the MomentumStrategy class.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 50: return None
    except:
        return None

    # 2. Initialize Strategy (Standard Defaults)
    strategy = MomentumStrategy(timeframe=timeframe, ema_period=20, rsi_period=14, rsi_limit=55)
    
    # Pre-calculate indicators for speed (Optimization)
    # The strategy usually does this inside get_signal, but for backtest speed we do it once
    df['ema'] = ta.ema(df['close'], length=20)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    position = 0
    entry_price = 0.0
    trades = 0
    wins = 0
    
    current_capital = capital
    
    # 3. Event Loop
    # We start at index 50 to allow indicators to warm up
    for i in range(50, len(df)):
        row = df.iloc[i]
        
        # Extract Signal
        # We manually check logic here to mimic the strategy's 'get_signal' 
        # because calling the full function 50,000 times is slower in Python loops.
        # This Logic MATCHES strategies/momentum_strategy.py
        
        # BUY LOGIC
        if position == 0:
            if row['close'] > row['ema'] and row['rsi'] > 55:
                # Buy Max Shares
                qty = int(current_capital / row['close'])
                if qty > 0:
                    position = qty
                    entry_price = row['close']
                    current_capital -= (qty * entry_price)
        
        # SELL LOGIC
        elif position > 0:
            # Exit if price drops below EMA (Trend break)
            # OR RSI gets too weak (< 45)
            if row['close'] < row['ema'] or row['rsi'] < 45:
                exit_price = row['close']
                pnl = (exit_price - entry_price) * position
                
                current_capital += (position * exit_price)
                
                trades += 1
                if pnl > 0: wins += 1
                position = 0

    # 4. Final Calculation
    # Close any open position at the last candle
    if position > 0:
        final_price = df.iloc[-1]['close']
        current_capital += (position * final_price)
        # Note: We don't count this as a 'trade' for win-rate stats strictly, 
        # but we add the equity value.

    total_pnl = current_capital - capital
    win_rate = (wins / trades * 100) if trades > 0 else 0.0
    return_pct = (total_pnl / capital) * 100

    return {
        "symbol": symbol,
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(current_capital, 2),
        "trades": trades,
        "win_rate": round(win_rate, 2),
        "return_pct": round(return_pct, 2)
    }
