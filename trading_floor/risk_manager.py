import math

class RiskManager:
    """
    Handles Position Sizing and Risk Constraints for StatArb.
    Implements Beta Neutrality with Lot Size constraints (Phase 5).
    """
    def __init__(self, capital_per_pair=100000, max_margin_usage=0.8):
        self.capital_per_pair = capital_per_pair
        self.max_margin_usage = max_margin_usage
        
    def calculate_sizing(self, price_y, price_x, beta, lot_y: int = 1, lot_x: int = 1):
        """
        Phase 5: Beta-Neutral Position Sizing (Seesaw Method)
        
        Formula:
        1. Start with the BIGGER lot size (Stock A)
        2. Stock_B_shares = Stock_A_lot_size ÷ Beta
        3. Round to actual lot sizes
        4. Scale by available capital
        """
        if price_y <= 0 or price_x <= 0 or beta == 0:
            return 0, 0
        
        beta = abs(beta)
        
        # STEP 1: Identify the bigger lot size
        if lot_x >= lot_y:
            # X has bigger lot - start with 1 lot of X
            required_y_shares = lot_x / beta
            lots_y_base = max(1, round(required_y_shares / lot_y))
            lots_x_base = 1
        else:
            # Y has bigger lot - start with 1 lot of Y  
            required_x_shares = lot_y * beta
            lots_x_base = max(1, round(required_x_shares / lot_x))
            lots_y_base = 1
        
        # STEP 2: Calculate margin and scale by capital
        usable_capital = self.capital_per_pair * self.max_margin_usage
        margin_per_lot_y = price_y * lot_y * 0.15
        margin_per_lot_x = price_x * lot_x * 0.15
        
        margin_per_base_pair = (margin_per_lot_y * lots_y_base) + (margin_per_lot_x * lots_x_base)
        num_pairs = max(1, int(usable_capital / margin_per_base_pair))
        
        # Final lot counts
        lots_y = lots_y_base * num_pairs
        lots_x = lots_x_base * num_pairs
        
        # Final quantities
        qty_y = lots_y * lot_y
        qty_x = lots_x * lot_x
        
        # STEP 3: Validate mismatch
        required_x_total = qty_y * beta
        mismatch_shares = qty_x - required_x_total
        mismatch_pct = abs(mismatch_shares) / required_x_total * 100 if required_x_total > 0 else 0
        
        self._last_mismatch = {
            'required_x': required_x_total,
            'actual_x': qty_x,
            'difference': mismatch_shares,
            'mismatch_pct': mismatch_pct
        }
        
        # Check mismatch threshold
        if mismatch_pct > 50.0:
            print(f"      🚫 BLOCKED: Mismatch {mismatch_pct:.0f}% > 50%")
            return 0, 0
        
        # Verify margin
        total_margin = (margin_per_lot_y * lots_y) + (margin_per_lot_x * lots_x)
        if total_margin > usable_capital:
            print(f"      ⚠️ Margin exceeded: ₹{total_margin:,.0f} > ₹{usable_capital:,.0f}")
            return 0, 0
        
        # Sizing calculated silently for compact mode
        
        return qty_y, qty_x
    
    def get_spot_adjustment(self) -> dict:
        """
        Returns spot market adjustment recommendation if mismatch > 5%.
        Call after calculate_sizing().
        """
        if not hasattr(self, '_last_mismatch'):
            return {}
        
        mismatch = self._last_mismatch
        if mismatch['mismatch_pct'] > 5.0:
            return {
                'needed': True,
                'shares': int(abs(mismatch['difference'])),
                'direction': 'BUY' if mismatch['difference'] < 0 else 'SELL',
                'mismatch_pct': mismatch['mismatch_pct']
            }
        return {'needed': False}
    
    def _extract_symbol(self, price):
        """Placeholder - actual symbol passed separately in engine."""
        return None

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
        # Exit when Z-Score reaches ±1.0 or below (per checklist Phase 8)
        if abs(current_z) <= 1.0:
            return True, f"Mean Reversion Achieved (Z={current_z:.2f})"
        return False, ""
