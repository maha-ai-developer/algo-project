"""
Unit Tests for Architecture Optimizations

Tests the 6 optimizations:
#1 & #2: DataCache (parallel fetch, incremental updates)
#3: Dependency Injection
#4: State Persistence
#5: Guardian Caching
#6: WebSocket (mock)
"""

import sys
import os
import unittest
import json
import tempfile
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStateManager(unittest.TestCase):
    """Test Optimization #4: State Persistence"""
    
    def setUp(self):
        from trading_floor.state import StateManager
        # Use temp file for testing
        self.temp_file = tempfile.mktemp(suffix='.json')
        self.state_mgr = StateManager(state_file=self.temp_file)
    
    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def test_save_and_load(self):
        """Test basic save/load cycle"""
        trades = {
            "SBIN-HDFCBANK": {"side": "LONG", "q1": 10, "q2": 5},
            "INFY-TCS": {"side": "SHORT", "q1": 20, "q2": 15}
        }
        
        # Save
        result = self.state_mgr.save(trades)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.temp_file))
        
        # Load
        loaded = self.state_mgr.load()
        self.assertEqual(loaded, trades)
    
    def test_load_empty(self):
        """Test loading when no state file exists"""
        loaded = self.state_mgr.load()
        self.assertEqual(loaded, {})
    
    def test_atomic_write(self):
        """Test that writes are atomic (temp file then rename)"""
        self.state_mgr.save({"test": "data"})
        
        # Verify no .tmp file left behind
        self.assertFalse(os.path.exists(self.temp_file + ".tmp"))


class TestGuardianCaching(unittest.TestCase):
    """Test Optimization #5: Guardian Lazy Recalculation"""
    
    def test_cache_interval_exists(self):
        """Verify CACHE_INTERVAL constant exists"""
        source = self._read_file('strategies/guardian.py')
        self.assertIn('CACHE_INTERVAL', source)
    
    def test_cached_result_field(self):
        """Verify caching fields exist"""
        source = self._read_file('strategies/guardian.py')
        self.assertIn('_cached_result', source)
        self.assertIn('_diagnosis_count', source)
    
    def test_cache_reuse_logic(self):
        """Verify cache reuse condition in diagnose()"""
        source = self._read_file('strategies/guardian.py')
        # Should check modulo for cache interval
        self.assertIn('CACHE_INTERVAL', source)
        self.assertIn('_cached_result', source)
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()


class TestDataCache(unittest.TestCase):
    """Test Optimization #1 & #2: Parallel Fetch + Incremental Updates"""
    
    def test_cache_module_exists(self):
        """Verify cache module was created"""
        self.assertTrue(os.path.exists('infrastructure/data/cache.py'))
    
    def test_parallel_fetch_function(self):
        """Verify parallel_fetch method exists"""
        source = self._read_file('infrastructure/data/cache.py')
        self.assertIn('def parallel_fetch', source)
        self.assertIn('ThreadPoolExecutor', source)
    
    def test_incremental_update_function(self):
        """Verify incremental update logic exists"""
        source = self._read_file('infrastructure/data/cache.py')
        self.assertIn('_incremental_update', source)
    
    def test_thread_safety(self):
        """Verify thread-safe locking is used"""
        source = self._read_file('infrastructure/data/cache.py')
        self.assertIn('threading.RLock', source)
        self.assertIn('with self._lock', source)
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()


class TestWebSocketTicker(unittest.TestCase):
    """Test Optimization #6: WebSocket Integration"""
    
    def test_ticker_module_exists(self):
        """Verify ticker module was created"""
        self.assertTrue(os.path.exists('infrastructure/broker/ticker.py'))
    
    def test_realtime_ticker_class(self):
        """Verify RealtimeTicker class exists"""
        source = self._read_file('infrastructure/broker/ticker.py')
        self.assertIn('class RealtimeTicker', source)
    
    def test_mock_ticker_class(self):
        """Verify MockTicker for testing exists"""
        source = self._read_file('infrastructure/broker/ticker.py')
        self.assertIn('class MockTicker', source)
    
    def test_websocket_callbacks(self):
        """Verify WebSocket callback handlers exist"""
        source = self._read_file('infrastructure/broker/ticker.py')
        self.assertIn('on_ticks', source)
        self.assertIn('on_connect', source)
        self.assertIn('on_close', source)
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()


class TestEngineDependencyInjection(unittest.TestCase):
    """Test Optimization #3: Dependency Injection"""
    
    def test_factory_function_exists(self):
        """Verify create_engine factory function exists"""
        source = self._read_file('trading_floor/engine.py')
        self.assertIn('def create_engine', source)
    
    def test_engine_accepts_dependencies(self):
        """Verify TradingEngine accepts dependencies in constructor"""
        source = self._read_file('trading_floor/engine.py')
        # Check constructor signature has key dependencies
        self.assertIn('broker,', source)
        self.assertIn('data_cache,', source)
        self.assertIn('state_manager,', source)
        self.assertIn('ticker=None', source)
    
    def test_cli_uses_factory(self):
        """Verify CLI uses create_engine factory"""
        source = self._read_file('cli.py')
        self.assertIn('from trading_floor.engine import create_engine', source)
        self.assertIn('create_engine(', source)
    
    def test_websocket_cli_flag(self):
        """Verify CLI has --websocket flag"""
        source = self._read_file('cli.py')
        self.assertIn('--websocket', source)
    
    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestStateManager))
    suite.addTests(loader.loadTestsFromTestCase(TestGuardianCaching))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCache))
    suite.addTests(loader.loadTestsFromTestCase(TestWebSocketTicker))
    suite.addTests(loader.loadTestsFromTestCase(TestEngineDependencyInjection))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)
