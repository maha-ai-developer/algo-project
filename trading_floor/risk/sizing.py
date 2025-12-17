def calculate_kelly_percentage(win_rate, reward_to_risk):
    """
    Calculates Kelly % = W - [(1-W)/R]
    Ref: Chapter 14, Risk Management & Trading Psychology
    """
    if reward_to_risk == 0: return 0
    kelly_pct = win_rate - ((1 - win_rate) / reward_to_risk)
    
    # Return 0 if negative (don't trade if edge is negative)
    return max(0, kelly_pct)

def get_optimal_quantity(equity, max_risk_pct, entry_price, stop_loss_price, win_rate=0.5, reward_risk=2.0):
    """
    Applies the 3-Step Optimal Strategy:
    1. Constraint: Max Risk % (e.g. 2%)
    2. Optimize: Scale by Kelly %
    3. Sizing: Calculate Quantity based on Risk per Share
    """
    # 1. Constraint: Calculate Hard Max Loss (e.g. 2% of 15,000 = 300)
    max_loss_allowed = equity * max_risk_pct

    # 2. Optimize: Calculate Kelly Score (e.g. 0.25)
    # Note: We use defaults W=0.5, R=2.0 for now. 
    # As you trade more, you can update these with real stats from your DB.
    kelly_factor = calculate_kelly_percentage(win_rate, reward_risk)

    # 3. Scale Risk: Apply Kelly to the Max Loss Constraint
    # (e.g. 300 * 0.25 = 75 Risk)
    optimized_risk_amount = max_loss_allowed * kelly_factor

    # 4. Quantity Calculation
    risk_per_share = abs(entry_price - stop_loss_price)
    
    # Avoid division by zero
    if risk_per_share <= 0.05: 
        risk_per_share = entry_price * 0.01 # Fallback to 1% width

    qty = int(optimized_risk_amount / risk_per_share)
    
    # Logging the math for transparency
    print(f"   ⧖ Risk Math: Equity ₹{equity} | MaxRisk ₹{max_loss_allowed:.1f} | Kelly {kelly_factor:.2f}")
    print(f"   ⧖ Optimized Risk: ₹{optimized_risk_amount:.1f} | Risk/Share ₹{risk_per_share:.1f} -> Qty {qty}")

    return max(1, qty) # Always return at least 1 to test
