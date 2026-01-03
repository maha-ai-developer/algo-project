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
        # Updated to match paper-maharajan.md Section 8.4.1
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
    
    def test_walk_forward_method(self):
        """Verify walk-forward optimization method exists"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('def run_walk_forward', source)
        self.assertIn('train_window', source)
        self.assertIn('test_window', source)
    
    def test_half_life_calculation(self):
        """Verify half-life calculation using Ornstein-Uhlenbeck"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('_calculate_half_life_ou', source)
        self.assertIn('Ornstein-Uhlenbeck', source)
    
    def test_rolling_adf_check(self):
        """Verify rolling ADF validation for cointegration monitoring"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('_rolling_adf_check', source)


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


class TestBacktestRiskMetrics(unittest.TestCase):
    """Test risk metrics in backtest output (Checklist Gap Fill)"""
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def test_sharpe_ratio_calculated(self):
        """Verify Sharpe ratio is calculated in backtest"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('sharpe_ratio', source)
        self.assertIn('np.sqrt(252)', source)  # Annualization factor
    
    def test_max_drawdown_calculated(self):
        """Verify max drawdown is calculated"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('max_drawdown', source)
        self.assertIn('running_max', source)
        self.assertIn('np.maximum.accumulate', source)
    
    def test_profit_factor_calculated(self):
        """Verify profit factor is calculated"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('profit_factor', source)
        self.assertIn('gross_profit', source)
        self.assertIn('gross_loss', source)
    
    def test_equity_curve_tracked(self):
        """Verify equity curve is tracked for risk metrics"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('equity_curve', source)
    
    def test_train_test_split_function(self):
        """Verify train/test split function exists"""
        source = self._read_file('research_lab/backtest_pairs.py')
        self.assertIn('def split_data', source)
        self.assertIn('TRAIN_PCT', source)
        self.assertIn('VALIDATE_PCT', source)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestDocCompliance))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestRiskMetrics))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

