"""
Unit Tests for Backtest Optimizations

Tests the backtest_pairs.py v2.0 structure and doc compliance.
"""

import sys
import os
import unittest
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBacktestDocCompliance(unittest.TestCase):
    """Test alignment with paper-maharajan.md specifications"""
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def test_lookback_window(self):
        """Verify lookback is 250 days per docs"""
        source = self._read_file('research_lab/backtest_pairs.py')
        match = re.search(r'LOOKBACK_WINDOW\s*=\s*(\d+)', source)
        self.assertIsNotNone(match, "LOOKBACK_WINDOW should be defined")
        self.assertEqual(int(match.group(1)), 250, "Lookback should be 250 days per paper-maharajan.md")
    
    def test_z_exit_threshold(self):
        """Verify Z-exit is 0.5 per docs (not 0.0)"""
        source = self._read_file('research_lab/backtest_pairs.py')
        match = re.search(r'Z_EXIT_THRESHOLD\s*=\s*([0-9.]+)', source)
        self.assertIsNotNone(match, "Z_EXIT_THRESHOLD should be defined")
        self.assertEqual(float(match.group(1)), 0.5, "Z-exit should be ±0.5 per paper-maharajan.md")
    
    def test_max_holding_days(self):
        """Verify max holding period is 10 days per docs"""
        source = self._read_file('research_lab/backtest_pairs.py')
        match = re.search(r'MAX_HOLDING_DAYS\s*=\s*(\d+)', source)
        self.assertIsNotNone(match, "MAX_HOLDING_DAYS should be defined")
        self.assertEqual(int(match.group(1)), 10, "Max hold should be 10 days per paper-maharajan.md")
    
    def test_slippage_modeled(self):
        """Verify slippage is modeled"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('SLIPPAGE_PCT', source)
        self.assertIn('STT_PCT', source)
    
    def test_no_lookahead_bias(self):
        """Verify rolling window comment indicating no look-ahead"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('NO LOOK-AHEAD', source.upper())


class TestBacktestOptimizations(unittest.TestCase):
    """Test optimization features"""
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def test_progress_manager(self):
        """Verify progress persistence class exists"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('class BacktestProgressManager', source)
        self.assertIn('def load(self', source)
        self.assertIn('def save(self', source)
    
    def test_resume_parameter(self):
        """Verify resume functionality"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('resume:', source)
    
    def test_guardian_integration(self):
        """Verify Guardian is used for health monitoring"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('AssumptionGuardian', source)
        self.assertIn('guardian.diagnose()', source)
        self.assertIn('needs_recalibration', source)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestDocCompliance))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestOptimizations))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
