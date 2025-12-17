# core/candle_builder.py
from datetime import datetime
import pandas as pd

_TIMEFRAME_RULES = {
    "1m": "1T",
    "3m": "3T",
    "5m": "5T",
    "15m": "15T",
    "1h": "60T",
    "1d": "1D"
}

class CandleBuilder:
    """
    Multi-instrument candle builder.
    Maintains an OHLCV DataFrame per symbol and timeframe.
    """
    def __init__(self, timeframe="5m"):
        if timeframe not in _TIMEFRAME_RULES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.timeframe = timeframe
        self._rule = _TIMEFRAME_RULES[timeframe]
        # symbol -> DataFrame(index=datetime, columns=[open,high,low,close,volume])
        self._bars = {}

    def add_tick(self, symbol: str, price: float, volume: float, ts: datetime):
        if symbol not in self._bars:
            self._bars[symbol] = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                dtype=float
            )
        df = self._bars[symbol]
        tkey = pd.to_datetime(ts).floor(self._rule)

        if tkey not in df.index:
            df.loc[tkey, "open"] = price
            df.loc[tkey, "high"] = price
            df.loc[tkey, "low"] = price
            df.loc[tkey, "close"] = price
            df.loc[tkey, "volume"] = volume
        else:
            row = df.loc[tkey]
            row["high"] = max(row["high"], price)
            row["low"] = min(row["low"], price)
            row["close"] = price
            row["volume"] += volume
            df.loc[tkey] = row

        self._bars[symbol] = df

    def get_bars(self, symbol: str) -> pd.DataFrame:
        return self._bars.get(symbol, pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))

    def get_latest_bar(self, symbol: str):
        df = self.get_bars(symbol)
        if df.empty:
            return None, None
        return df.index[-1], df.iloc[-1]
