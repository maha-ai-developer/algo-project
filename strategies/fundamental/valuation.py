# strategies/fundamental/valuation.py

class DCFModel:
    """
    Discounted Cash Flow (DCF) Model.
    Source: Integrated Financial Modeling PDF.
    
    Logic:
    1. Project Free Cash Flow (FCFF) for next 5-10 years.
    2. Calculate WACC (Discount Rate).
    3. Calculate Terminal Value (Perpetual growth after year 10).
    4. Discount everything to Present Value (PV).
    5. Apply Margin of Safety (10% band).
    """

    def __init__(self, risk_free_rate=0.07, market_return=0.12):
        """
        risk_free_rate: India 10Y Bond Yield (approx 7%)
        market_return: Nifty Historical Return (approx 12%)
        """
        self.rf = risk_free_rate
        self.rm = market_return

    def calculate_cost_of_equity(self, beta):
        """CAPM Model: Ke = Rf + Beta * (Rm - Rf)"""
        return self.rf + beta * (self.rm - self.rf)

    def calculate_wacc(self, beta, equity_weight, debt_weight, cost_of_debt, tax_rate=0.25):
        """
        Weighted Average Cost of Capital.
        """
        ke = self.calculate_cost_of_equity(beta)
        kd_after_tax = cost_of_debt * (1 - tax_rate)
        
        wacc = (ke * equity_weight) + (kd_after_tax * debt_weight)
        return wacc

    def get_intrinsic_value(self, free_cash_flows, terminal_growth_rate, wacc, shares_outstanding, net_debt):
        """
        Calculates Fair Share Price.
        
        Args:
            free_cash_flows (list): Projected FCFF for next N years.
            terminal_growth_rate (float): Usually 3-4% (Inflation rate).
            wacc (float): Discount rate.
            shares_outstanding (int): Total count of shares.
            net_debt (float): Total Debt - Cash.
        """
        # 1. Sum of Present Values of Projected Cash Flows
        present_value_fcf = 0
        for year, cash_flow in enumerate(free_cash_flows, 1):
            present_value_fcf += cash_flow / ((1 + wacc) ** year)

        # 2. Terminal Value (Value beyond projection period)
        last_fcf = free_cash_flows[-1]
        terminal_value = (last_fcf * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
        
        # Discount Terminal Value to today
        pv_terminal_value = terminal_value / ((1 + wacc) ** len(free_cash_flows))

        # 3. Enterprise Value
        enterprise_value = present_value_fcf + pv_terminal_value

        # 4. Equity Value (EV - Debt)
        equity_value = enterprise_value - net_debt

        # 5. Fair Price per Share
        fair_price = equity_value / shares_outstanding

        # 6. Apply Margin of Safety (10% as per PDF)
        buy_price = fair_price * 0.90 

        return {
            "fair_value": round(fair_price, 2),
            "buy_price": round(buy_price, 2),
            "wacc": round(wacc * 100, 2),
            "margin_of_safety": "10%"
        }
