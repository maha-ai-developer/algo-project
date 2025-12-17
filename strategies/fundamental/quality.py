# strategies/fundamental/quality.py

class QualityCheck:
    """
    Qualitative & Ratio Analysis.
    Source: Fundamental Analysis PDF.
    
    Filters:
    1. Growth > 20% (Sales/Profit)
    2. ROE > 15-20% (Efficiency)
    3. Debt/Equity < 1.0 (Safety)
    """

    def __init__(self):
        # Thresholds defined in the PDF
        self.min_growth = 0.15      # 15% Growth
        self.min_roe = 0.15         # 15% ROE
        self.max_debt_equity = 1.0  # Max Debt 1:1

    def evaluate(self, financials):
        """
        Input: Dictionary containing key financial ratios.
        Returns: Score (0-10) and Status (Pass/Fail).
        """
        score = 0
        reasons = []

        # 1. Growth Check
        if financials.get('sales_growth', 0) > self.min_growth:
            score += 3
        else:
            reasons.append(f"Low Sales Growth ({financials.get('sales_growth')*100}%)")

        if financials.get('profit_growth', 0) > self.min_growth:
            score += 3
        else:
            reasons.append("Low Profit Growth")

        # 2. Efficiency Check (ROE)
        if financials.get('roe', 0) > self.min_roe:
            score += 2
        else:
            reasons.append("Low ROE")

        # 3. Safety Check (Debt)
        if financials.get('debt_to_equity', 2.0) < self.max_debt_equity:
            score += 2
        else:
            reasons.append("High Debt")

        # Final Verdict
        is_investible = score >= 7  # Must score at least 7/10
        
        return {
            "score": score,
            "status": "INVESTIBLE" if is_investible else "AVOID",
            "flags": reasons
        }
