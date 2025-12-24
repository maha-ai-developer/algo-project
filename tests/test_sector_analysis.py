"""
Unit Tests for Sector Analysis Optimizations

Tests the sector_analysis.py v2.0 structure using source code analysis.
"""

import sys
import os
import unittest
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSectorAnalysisOptimizations(unittest.TestCase):
    """Test that all optimizations are present"""
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def test_parallel_processing(self):
        """Verify ThreadPoolExecutor is used"""
        source = self._read_file('research_lab/sector_analysis.py')
        self.assertIn('ThreadPoolExecutor', source)
        self.assertIn('as_completed', source)
        self.assertIn('max_workers', source)
    
    def test_cache_class(self):
        """Verify SectorCache class exists"""
        source = self._read_file('research_lab/sector_analysis.py')
        self.assertIn('class SectorCache', source)
        self.assertIn('def get(self', source)
        self.assertIn('def set(self', source)
    
    def test_retry_logic(self):
        """Verify retry with exponential backoff"""
        source = self._read_file('research_lab/sector_analysis.py')
        self.assertIn('fetch_sector_with_retry', source)
        self.assertIn('RETRY_ATTEMPTS', source)
        self.assertIn('2 ** attempt', source)
    
    def test_progress_class(self):
        """Verify SectorProgressManager exists"""
        source = self._read_file('research_lab/sector_analysis.py')
        self.assertIn('class SectorProgressManager', source)
        self.assertIn('def load(self', source)
        self.assertIn('def save(self', source)
    
    def test_thread_safety(self):
        """Verify thread-safe locking"""
        source = self._read_file('research_lab/sector_analysis.py')
        self.assertIn('threading.Lock', source)
        self.assertIn('self._lock', source)
    
    def test_config_constants(self):
        """Verify configuration constants"""
        source = self._read_file('research_lab/sector_analysis.py')
        self.assertIn('MAX_WORKERS', source)
        self.assertIn('RETRY_ATTEMPTS', source)
        self.assertIn('CACHE_EXPIRY_HOURS', source)
        self.assertIn('API_DELAY', source)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSectorAnalysisOptimizations)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
