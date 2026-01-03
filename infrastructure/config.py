import os
import json

# =========================================================
# 1. PATH CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define Data Directories (Legacy - kept for backward compatibility)
DATA_DIR = os.path.join(BASE_DIR, "data", "historical")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
UNIVERSE_DIR = os.path.join(BASE_DIR, "data", "universe")

# NEW: Structured Data Paths (Research-backed separation)
# ───────────────────────────────────────────────────────
# Pair Selection: Short-term spot data for scanning cointegrated pairs
PAIR_SELECTION_DIR = os.path.join(BASE_DIR, "data", "pair_selection", "spot")

# Backtesting: Long-term data for robust statistical validation
BACKTEST_SPOT_DIR = os.path.join(BASE_DIR, "data", "historical", "spot")
BACKTEST_FUTURES_DIR = os.path.join(BASE_DIR, "data", "historical", "futures")

# Data Duration Settings (per research benchmarks)
PAIR_SELECTION_DAYS = 365   # 1 year for pair scanning
BACKTEST_SPOT_DAYS = 750    # 3 years for robust backtesting

# Define Config Directory
CONFIG_DIR = os.path.join(BASE_DIR, "infrastructure", "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
SESSION_FILE = os.path.join(CONFIG_DIR, "kite_session.json")

# Ensure critical directories exist
for d in [DATA_DIR, ARTIFACTS_DIR, LOG_DIR, CACHE_DIR, UNIVERSE_DIR, CONFIG_DIR,
          PAIR_SELECTION_DIR, BACKTEST_SPOT_DIR, BACKTEST_FUTURES_DIR]:
    os.makedirs(d, exist_ok=True)

# --- KEY FILES ---
UNIVERSE_FILE = os.path.join(ARTIFACTS_DIR, "symbols.txt") 
PAIRS_CANDIDATES_FILE = os.path.join(ARTIFACTS_DIR, "pairs_candidates.json") 
PAIRS_DOWNLOAD_FILE = os.path.join(ARTIFACTS_DIR, "pairs.txt")
FUNDAMENTAL_FILE = os.path.join(ARTIFACTS_DIR, "fundamental_analysis.csv")
SECTOR_REPORT_FILE = os.path.join(ARTIFACTS_DIR, "sector_report.csv") 
MOMENTUM_CONFIG = os.path.join(ARTIFACTS_DIR, "momentum_config.json")
PAIRS_CONFIG = os.path.join(ARTIFACTS_DIR, "pairs_config.json")

# --- GLOBAL SETTINGS ---
TIMEFRAME = "5m"
PAIR_SCAN_DAYS = 365
PAIR_CORRELATION_MIN = 0.8
PAIR_PVALUE_MAX = 0.05

# --- AI MODEL SETTINGS ---
# Options:
# - gemini-3-pro-preview
# - gemini-3-flash-preview
# - gemini-2.5-flash
# - gemini-2.5-flash-lite
# - gemini-2.5-pro
# - gemini-2.0-flash
# - gemini-2.0-flash-lite
GENAI_MODEL = "gemini-3-flash-preview"  # <--- SWITCHED: Higher RPD limit than pro

# =========================================================
# 2. CREDENTIALS LOADER
# =========================================================
def load_credentials():
    if not os.path.exists(CONFIG_FILE):
        template = {
            "kite": {
                "api_key": "YOUR_KITE_KEY",
                "api_secret": "YOUR_KITE_SECRET",
                "access_token": "",
                "redirect_url": "http://127.0.0.1:5000/login"
            },
            "genai": {
                "api_key": "YOUR_GOOGLE_GENAI_KEY"
            }
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(template, f, indent=4)
        print(f"⚠️  Created template at {CONFIG_FILE}. Please fill your API keys!")
        return None, None, None, None

    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        
        kite = data.get("kite", {})
        genai = data.get("genai", {})
        
        return (
            kite.get("api_key"),
            kite.get("api_secret"),
            kite.get("access_token"),
            genai.get("api_key")
        )
    except Exception as e:
        print(f"❌ Error reading config.json: {e}")
        return None, None, None, None

def save_access_token(access_token):
    if not os.path.exists(CONFIG_FILE): return
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        if "kite" not in data: data["kite"] = {}
        data["kite"]["access_token"] = access_token
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"❌ Error saving token: {e}")

API_KEY, API_SECRET, ACCESS_TOKEN, GENAI_API_KEY = load_credentials()
DB_URL = f"sqlite:///{os.path.join(DATA_DIR, 'trades.db')}"
