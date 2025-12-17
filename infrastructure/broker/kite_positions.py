from infrastructure.broker.kite_auth import get_kite

def fetch_account_snapshot():
    """
    Returns: (profile, margins, holdings, positions)
    """
    try:
        kite = get_kite()
        
        # 1. Profile & Margins
        profile = kite.profile()
        margins = kite.margins(segment="equity") # 'equity' or 'commodity'

        # 2. Holdings (Long Term)
        holdings = kite.holdings()

        # 3. Positions (Intraday/F&O)
        # Returns {'net': [...], 'day': [...]}
        positions = kite.positions()

        return profile, margins, holdings, positions

    except Exception as e:
        print(f"[Broker] Snapshot failed: {e}")
        return {}, {}, [], {}

def get_open_positions():
    """
    Returns list of open positions (Net quantity != 0)
    """
    _, _, _, positions_data = fetch_account_snapshot()
    if not positions_data or 'net' not in positions_data:
        return []
    
    # Filter for non-zero quantity
    return [p for p in positions_data['net'] if p['quantity'] != 0]
