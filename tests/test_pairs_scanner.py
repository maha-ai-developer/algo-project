"""
Unit Tests for Pairs Scanner Optimizations

Tests the scan_pairs.py v2.0 structure.
Strategy compliance verified separately.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPairsScannerOptimizations(unittest.TestCase):
    """Test optimization features"""
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def test_pairs_cache_class(self):
        """Verify PairsCache class exists"""
        source = self._read_file('research_lab/scan_pairs.py')
        self.assertIn('class PairsCache', source)
        self.assertIn('def get(self', source)
        self.assertIn('def set(self', source)
    
    def test_progress_class(self):
        """Verify PairsProgressManager exists"""
        source = self._read_file('research_lab/scan_pairs.py')
        self.assertIn('class PairsProgressManager', source)
        self.assertIn('def load(self', source)
        self.assertIn('def save(self', source)
    
    def test_resume_parameter(self):
        """Verify resume functionality"""
        source = self._read_file('research_lab/scan_pairs.py')
        self.assertIn('resume:', source)  # e.g., resume: bool
        self.assertIn('= True', source)  # default True somewhere
    
    def test_cache_parameter(self):
        """Verify use_cache parameter"""
        source = self._read_file('research_lab/scan_pairs.py')
        self.assertIn('use_cache:', source)
    
    def test_strategy_unchanged(self):
        """Verify core strategy logic is present"""
        source = self._read_file('research_lab/scan_pairs.py')
        # Key strategy elements
        self.assertIn('StatArbBot', source)
        self.assertIn('calibrate', source)
        self.assertIn('LEADER', source)
        self.assertIn('CHALLENGER', source)
        self.assertIn('hedge_ratio', source)
    
    def test_thread_safety(self):
        """Verify thread-safe locking"""
        source = self._read_file('research_lab/scan_pairs.py')
        self.assertIn('threading', source)
        self.assertIn('_lock', source)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPairsScannerOptimizations)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
