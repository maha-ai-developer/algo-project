import math

class RiskManager:
    """
    Handles Position Sizing and Risk Constraints for StatArb.
    """
    def __init__(self, capital_per_pair=100000, max_margin_usage=0.8):
        self.capital_per_pair = capital_per_pair
        self.max_margin_usage = max_margin_usage
        
    def calculate_sizing(self, price_y, price_x, beta):
        """
        Calculates Beta-Neutral Position Sizes.
        Goal: Value_X should hedge Value_Y based on Beta.
        """
        if price_y <= 0 or price_x <= 0:
            return 0, 0

        # Formula:
        # Allocation Y = Capital / (1 + Beta)  <-- Approximation for balance
        # But commonly we split capital 50/50 notionally or fix leg 1.
        
        # Professional Approach: Fix Capital for Y, Scale X by Beta
        # This ensures the 'Hedge' is sized correctly relative to the 'Asset'.
        
        allocation_y = self.capital_per_pair / 2
        
        qty_y = int(allocation_y / price_y)
        
        # Value_X should be Value_Y * Beta
        # Qty_X * Price_X = (Qty_Y * Price_Y) * Beta
        # Qty_X = (Qty_Y * Price_Y * Beta) / Price_X
        
        value_y = qty_y * price_y
        qty_x = int((value_y * abs(beta)) / price_x)
        
        return qty_y, qty_x

    def check_stop_loss(self, current_z, stop_z_threshold):
        """
        Statistical Stop Loss.
        Returns True if Z-Score has blown out (Structural Break).
        """
        if abs(current_z) >= stop_z_threshold:
            return True, f"Z-Score Breach ({current_z:.2f} > {stop_z_threshold})"
        return False, ""

    def check_take_profit(self, current_z, exit_z_threshold=0.0):
        """
        Mean Reversion Target.
        Returns True if Z-Score has crossed zero (Fair Value).
        """
        # We use a small band (e.g., 0.0 to 0.5) to ensure we actually fill
        if abs(current_z) <= 0.5:
            return True, f"Mean Reversion Achieved (Z={current_z:.2f})"
        return False, ""
