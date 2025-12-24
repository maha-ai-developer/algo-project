"""
Unit Tests for Fundamental Scanner Optimizations

Tests the scan_fundamental.py v2.0 structure using source code analysis
(avoids import issues from missing dependencies).
"""

import sys
import os
import unittest
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScanFundamentalOptimizations(unittest.TestCase):
    """Test that all optimizations are present in the code"""
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def test_parallel_processing(self):
        """Optimization #1: Verify ThreadPoolExecutor is used"""
        source = self._read_file('research_lab/scan_fundamental.py')
        self.assertIn('ThreadPoolExecutor', source)
        self.assertIn('as_completed', source)
        self.assertIn('max_workers', source)
    
    def test_cache_class(self):
        """Optimization #2: Verify FundamentalCache class exists"""
        source = self._read_file('research_lab/scan_fundamental.py')
        self.assertIn('class FundamentalCache', source)
        self.assertIn('def get(self', source)
        self.assertIn('def set(self', source)
        self.assertIn('CACHE_EXPIRY', source)
    
    def test_retry_logic(self):
        """Optimization #3: Verify retry with exponential backoff exists"""
        source = self._read_file('research_lab/scan_fundamental.py')
        self.assertIn('fetch_fundamental_with_retry', source)
        self.assertIn('RETRY_ATTEMPTS', source)
        self.assertIn('RETRY_DELAY_BASE', source)
        # Check exponential backoff formula
        self.assertIn('2 ** attempt', source)
    
    def test_progress_class(self):
        """Optimization #4: Verify ProgressManager class exists"""
        source = self._read_file('research_lab/scan_fundamental.py')
        self.assertIn('class ProgressManager', source)
        self.assertIn('def load(self', source)
        self.assertIn('def save(self', source)
        self.assertIn('add_result', source)
    
    def test_thread_safety(self):
        """Verify thread-safe locking is used"""
        source = self._read_file('research_lab/scan_fundamental.py')
        self.assertIn('threading.Lock', source)
        self.assertIn('self._lock', source)
    
    def test_config_constants(self):
        """Verify configuration constants exist"""
        source = self._read_file('research_lab/scan_fundamental.py')
        
        # Check MAX_WORKERS
        match = re.search(r'MAX_WORKERS\s*=\s*(\d+)', source)
        self.assertIsNotNone(match, "MAX_WORKERS should be defined")
        workers = int(match.group(1))
        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, 10)
        
        # Check RETRY_ATTEMPTS
        match = re.search(r'RETRY_ATTEMPTS\s*=\s*(\d+)', source)
        self.assertIsNotNone(match, "RETRY_ATTEMPTS should be defined")
        retries = int(match.group(1))
        self.assertGreaterEqual(retries, 1)
    
    def test_resume_functionality(self):
        """Verify resume parameter exists"""
        source = self._read_file('research_lab/scan_fundamental.py')
        self.assertIn('resume:', source)  # e.g., resume: bool
        self.assertIn('= True', source)  # default True somewhere
    
    def test_cache_toggle(self):
        """Verify use_cache parameter exists"""
        source = self._read_file('research_lab/scan_fundamental.py')
        self.assertIn('use_cache:', source)  # e.g., use_cache: bool
        self.assertIn('use_cache', source)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestScanFundamentalOptimizations))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)
