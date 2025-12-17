# strategies/base_strategy.py

class BaseStrategy:
    """
    Interface that all strategies must implement.
    """
    def __init__(self, name):
        self.name = name

    def generate_signal(self, data):
        """
        Analyzes data and returns a signal.
        
        Args:
            data: DataFrame or Dict of DataFrames
            
        Returns:
            dict: {
                "signal": "BUY" | "SELL" | "EXIT" | "NONE",
                "reason": str,
                "price": float,
                "metadata": {}
            }
        """
        raise NotImplementedError("Strategy must implement generate_signal()")
