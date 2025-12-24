"""
State Persistence Module - Optimization #4

Persists active trades to JSON file so engine can recover after restart.
Prevents losing track of open positions on crashes/restarts.
"""

import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional

import infrastructure.config as config


class StateManager:
    """
    Thread-safe state persistence manager.
    
    Saves and loads trading state (active_trades, etc.) to JSON.
    Auto-saves on every state change for crash recovery.
    """
    
    def __init__(self, state_file: Optional[str] = None):
        """
        Args:
            state_file: Path to state JSON file. Defaults to data/engine_state.json
        """
        self.state_file = state_file or os.path.join(config.DATA_DIR, "engine_state.json")
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {
            "active_trades": {},
            "last_updated": None,
            "version": "1.0"
        }
    
    def load(self) -> Dict[str, Any]:
        """
        Load state from disk.
        
        Returns:
            Dict of active trades (empty dict if no state file)
        """
        with self._lock:
            if not os.path.exists(self.state_file):
                print("   📂 No existing state file. Starting fresh.")
                return {}
            
            try:
                with open(self.state_file, 'r') as f:
                    self._state = json.load(f)
                
                active_trades = self._state.get("active_trades", {})
                last_updated = self._state.get("last_updated", "unknown")
                print(f"   📂 Loaded state: {len(active_trades)} active trades (last: {last_updated})")
                return active_trades
                
            except json.JSONDecodeError as e:
                print(f"   ⚠️ Corrupt state file, starting fresh: {e}")
                return {}
            except Exception as e:
                print(f"   ⚠️ Failed to load state: {e}")
                return {}
    
    def save(self, active_trades: Dict[str, Any]) -> bool:
        """
        Save state to disk.
        
        Args:
            active_trades: Dict of active trades to persist
            
        Returns:
            True if saved successfully
        """
        with self._lock:
            self._state["active_trades"] = active_trades
            self._state["last_updated"] = datetime.now().isoformat()
            
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
                
                # Write atomically (write to temp, then rename)
                temp_file = self.state_file + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(self._state, f, indent=2, default=str)
                
                os.replace(temp_file, self.state_file)
                return True
                
            except Exception as e:
                print(f"   ❌ Failed to save state: {e}")
                return False
    
    def update_trade(self, pair_key: str, trade_data: Dict[str, Any], all_trades: Dict[str, Any]):
        """
        Update a single trade and auto-save.
        
        Args:
            pair_key: Trade identifier (e.g., "SBIN-HDFCBANK")
            trade_data: Trade details
            all_trades: Full active_trades dict
        """
        self.save(all_trades)
    
    def remove_trade(self, pair_key: str, all_trades: Dict[str, Any]):
        """
        Remove a trade and auto-save.
        
        Args:
            pair_key: Trade identifier to remove
            all_trades: Full active_trades dict (already with trade removed)
        """
        self.save(all_trades)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current in-memory state."""
        with self._lock:
            return dict(self._state)
    
    def clear(self):
        """Clear state and delete file."""
        with self._lock:
            self._state = {
                "active_trades": {},
                "last_updated": None,
                "version": "1.0"
            }
            if os.path.exists(self.state_file):
                try:
                    os.remove(self.state_file)
                    print("   🗑️ State file cleared.")
                except Exception as e:
                    print(f"   ⚠️ Failed to delete state file: {e}")
