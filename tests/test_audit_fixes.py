"""
Unit Tests for Audit Fixes
Tests the critical compliance fixes identified in the codebase audit.
These tests are designed to run WITHOUT external dependencies (kiteconnect, pandas).
"""

import sys
import os
import unittest
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMarginFunctions(unittest.TestCase):
    """Test Fix #3: Margin check functions in sizing.py"""
    
    def setUp(self):
        from trading_floor.risk.sizing import check_margin_availability, get_futures_margin
        self.check_margin = check_margin_availability
        self.get_margin = get_futures_margin
    
    def test_margin_sufficient(self):
        """Test when equity exceeds margin + buffer"""
        is_ok, msg = self.check_margin(equity=100000, initial_margin=50000, m2m_buffer_pct=0.20)
        self.assertTrue(is_ok)
        self.assertIn("Margin OK", msg)
    
    def test_margin_insufficient(self):
        """Test when equity is below required margin"""
        is_ok, msg = self.check_margin(equity=50000, initial_margin=60000, m2m_buffer_pct=0.20)
        self.assertFalse(is_ok)
        self.assertIn("INSUFFICIENT", msg)
    
    def test_margin_exact_buffer(self):
        """Test edge case: equity exactly equals margin + 20% buffer"""
        # 50000 * 1.2 = 60000
        is_ok, msg = self.check_margin(equity=60000, initial_margin=50000, m2m_buffer_pct=0.20)
        self.assertTrue(is_ok)
    
    def test_futures_margin_calculation(self):
        """Test margin estimation for futures position"""
        # 100 qty * 1000 price * 15% = 15000
        margin = self.get_margin(symbol="NIFTY25JANFUT", qty=100, price=1000, margin_pct=0.15)
        self.assertEqual(margin, 15000)
    
    def test_futures_margin_custom_pct(self):
        """Test with custom margin percentage"""
        margin = self.get_margin(symbol="BANKNIFTY", qty=50, price=500, margin_pct=0.20)
        self.assertEqual(margin, 5000)  # 50 * 500 * 0.20


class TestPositionSizing(unittest.TestCase):
    """Test existing position sizing functions"""
    
    def test_kelly_percentage_positive_edge(self):
        from trading_floor.risk.sizing import calculate_kelly_percentage
        # W=0.6, R=2 -> Kelly = 0.6 - ((1-0.6)/2) = 0.6 - 0.2 = 0.4
        kelly = calculate_kelly_percentage(win_rate=0.6, reward_to_risk=2.0)
        self.assertAlmostEqual(kelly, 0.4, places=2)
    
    def test_kelly_percentage_negative_edge(self):
        from trading_floor.risk.sizing import calculate_kelly_percentage
        # W=0.3, R=1 -> Kelly = 0.3 - 0.7 = -0.4 -> capped at 0
        kelly = calculate_kelly_percentage(win_rate=0.3, reward_to_risk=1.0)
        self.assertEqual(kelly, 0)
    
    def test_optimal_quantity_calculation(self):
        from trading_floor.risk.sizing import get_optimal_quantity
        # Entry 100, SL 90 -> risk/share = 10
        # Equity 100000, max_risk 2% = 2000
        # Kelly with W=0.5, R=2 = 0.25
        # Optimized risk = 2000 * 0.25 = 500
        # Qty = 500 / 10 = 50
        qty = get_optimal_quantity(
            equity=100000, max_risk_pct=0.02,
            entry_price=100, stop_loss_price=90,
            win_rate=0.5, reward_risk=2.0
        )
        self.assertEqual(qty, 50)


class TestSourceCodeCompliance(unittest.TestCase):
    """
    Test source code compliance by reading files directly.
    This avoids import issues from missing dependencies.
    """
    
    def read_file(self, path):
        """Helper to read source file"""
        with open(path, 'r') as f:
            return f.read()
    
    def test_engine_uses_nrml(self):
        """Fix #1: Verify engine.py PRODUCT_TYPE is set to NRML"""
        source = self.read_file('trading_floor/engine.py')
        
        # Find the PRODUCT_TYPE assignment
        match = re.search(r'PRODUCT_TYPE\s*=\s*["\'](\w+)["\']', source)
        self.assertIsNotNone(match, "PRODUCT_TYPE assignment not found")
        self.assertEqual(match.group(1), "NRML", 
                        "PRODUCT_TYPE should be 'NRML' for futures overnight positions")
    
    def test_kite_orders_has_exchange_param(self):
        """Fix #2: Verify place_order has exchange parameter"""
        source = self.read_file('infrastructure/broker/kite_orders.py')
        
        # Check function signature includes exchange
        self.assertIn('exchange="NSE"', source,
                     "place_order should have exchange='NSE' default parameter")
        
        # Check NFO mapping exists
        self.assertIn('EXCHANGE_NFO', source,
                     "Should map to EXCHANGE_NFO for futures")
    
    def test_execution_has_stop_loss_function(self):
        """Fix #4: Verify place_stop_loss_order function exists"""
        source = self.read_file('trading_floor/execution.py')
        
        # Check function definition exists
        self.assertIn('def place_stop_loss_order', source,
                     "ExecutionHandler should have place_stop_loss_order method")
        
        # Check it uses SL-M order type
        self.assertIn('order_type="SL-M"', source,
                     "Stop loss should use SL-M order type for guaranteed fill")
    
    def test_gemini_schema_has_pe_roce(self):
        """Fix #5: Verify Gemini schema includes P/E and ROCE"""
        source = self.read_file('infrastructure/llm/client.py')
        
        # Check schema has pe_ratio
        self.assertIn('pe_ratio', source,
                     "Gemini schema should include pe_ratio field")
        
        # Check schema has roce_latest
        self.assertIn('roce_latest', source,
                     "Gemini schema should include roce_latest field")
    
    def test_sizing_has_margin_check(self):
        """Fix #3: Verify margin check functions exist"""
        source = self.read_file('trading_floor/risk/sizing.py')
        
        # Check function definitions exist
        self.assertIn('def check_margin_availability', source,
                     "sizing.py should have check_margin_availability function")
        
        self.assertIn('def get_futures_margin', source,
                     "sizing.py should have get_futures_margin function")
        
        # Check M2M buffer is used
        self.assertIn('m2m_buffer', source,
                     "Should include M2M buffer in margin check")


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMarginFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionSizing))
    suite.addTests(loader.loadTestsFromTestCase(TestSourceCodeCompliance))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)
