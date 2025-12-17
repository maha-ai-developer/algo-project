import pandas as pd
import numpy as np

class PairStrategy:
    def __init__(self, lookback=20, entry_std=2.0, exit_std=0.5):
        self.lookback = lookback
        self.entry_std = entry_std
        self.exit_std = exit_std

    def calculate_zscore(self, series_a, series_b):
        # Calculate the ratio (Spread)
        ratio = series_a / series_b
        
        # Calculate rolling mean and std deviation
        rolling_mean = ratio.rolling(window=self.lookback).mean()
        rolling_std = ratio.rolling(window=self.lookback).std()
        
        # Calculate Z-Score
        zscore = (ratio - rolling_mean) / rolling_std
        return zscore

    def generate_signal(self, df_a, df_b):
        """
        Expects two dataframes with 'close' columns.
        """
        # Align data
        df = pd.concat([df_a['close'], df_b['close']], axis=1).dropna()
        df.columns = ['leg1', 'leg2']
        
        if len(df) < self.lookback:
            return {'signal': 'WAIT', 'zscore': 0}

        z_series = self.calculate_zscore(df['leg1'], df['leg2'])
        current_z = z_series.iloc[-1]
        
        signal = "WAIT"
        # Logic: Mean Reversion
        # If Z > 2: Ratio is high (Leg1 Expensive, Leg2 Cheap) -> Short Spread
        if current_z > self.entry_std:
            signal = "SHORT_SPREAD" # Sell Leg1, Buy Leg2
            
        # If Z < -2: Ratio is low (Leg1 Cheap, Leg2 Expensive) -> Long Spread
        elif current_z < -self.entry_std:
            signal = "LONG_SPREAD"  # Buy Leg1, Sell Leg2
            
        # Exit Logic
        elif abs(current_z) < self.exit_std:
            signal = "EXIT"

        return {
            'signal': signal,
            'zscore': round(current_z, 2),
            'price_leg1': df['leg1'].iloc[-1],
            'price_leg2': df['leg2'].iloc[-1]
        }
