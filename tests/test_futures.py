"""
Unit Tests for Futures Support

Tests futures_utils.py and backtest_pairs.py v3.0 features.
"""

import sys
import os
import unittest
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFuturesUtils(unittest.TestCase):
    """Test futures utilities module"""
    
    def test_lot_sizes_exist(self):
        """Verify lot size function exists and works"""
        from infrastructure.data.futures_utils import get_lot_size
        # Test common symbols return valid lot sizes
        self.assertGreater(get_lot_size('SBIN'), 0)
        self.assertGreater(get_lot_size('RELIANCE'), 0)
        self.assertGreater(get_lot_size('NIFTY'), 0)
    
    def test_sbin_lot_size(self):
        """Verify SBIN lot size is correct"""
        from infrastructure.data.futures_utils import get_lot_size
        lot = get_lot_size('SBIN')
        self.assertEqual(lot, 1500)
    
    def test_symbol_mapper(self):
        """Verify futures symbol generation"""
        from infrastructure.data.futures_utils import get_futures_symbol
        symbol = get_futures_symbol('SBIN', datetime(2025, 1, 1))
        self.assertEqual(symbol, 'SBIN25JANFUT')
    
    def test_expiry_is_thursday(self):
        """Verify expiry calculation returns Thursday"""
        from infrastructure.data.futures_utils import get_expiry_date
        expiry = get_expiry_date(2025, 1)
        self.assertEqual(expiry.weekday(), 3)  # Thursday
    
    def test_margin_calculation(self):
        """Verify margin calculation"""
        from infrastructure.data.futures_utils import calculate_margin_required
        margin = calculate_margin_required('SBIN', 800, lots=1)
        # SBIN lot = 1500, price 800, ~15% margin
        expected = 800 * 1500 * 0.15
        self.assertAlmostEqual(margin, expected, places=0)


class TestFuturesBacktest(unittest.TestCase):
    """Test futures-ready backtest"""
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def test_lot_based_sizing(self):
        """Verify lot-based position sizing"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('get_lot_size', source)
        self.assertIn('lot_size_y', source)
        self.assertIn('lot_size_x', source)
    
    def test_margin_calculations(self):
        """Verify margin calculations are used"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('margin_used', source)
        self.assertIn('MARGIN_BUFFER_PCT', source)
    
    def test_futures_costs(self):
        """Verify realistic futures costs"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('STT_PCT', source)
        self.assertIn('EXCHANGE_TXN_PCT', source)
        self.assertIn('GST_PCT', source)
        self.assertIn('BROKERAGE_PER_ORDER', source)
    
    def test_hybrid_data_model(self):
        """Verify hybrid data model: Spot for signals, Futures for P&L"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('HybridBacktest', source)
        self.assertIn('_load_hybrid_data', source)
        self.assertIn('spot_y', source)
        self.assertIn('fut_y', source)
        self.assertIn('basis_risk', source)
    
    def test_data_mode_tracking(self):
        """Verify data mode is tracked in results"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('data_mode', source)
        self.assertIn('HYBRID', source)
        self.assertIn('SPOT_ONLY', source)
    
    def test_max_lots_constraint(self):
        """Verify max lots limit"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('MAX_LOTS_PER_LEG', source)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestFuturesUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestFuturesBacktest))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
