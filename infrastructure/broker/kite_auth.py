import sys
import os
from kiteconnect import KiteConnect

# Ensure root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import infrastructure.config as config

def get_kite():
    """
    Returns an authenticated KiteConnect instance.
    """
    # Reload config in case token was just updated
    api_key, api_secret, access_token, _ = config.load_credentials()

    if not api_key:
        raise Exception("❌ API Key missing in config/config.json")

    kite = KiteConnect(api_key=api_key)

    if access_token:
        kite.set_access_token(access_token)
    
    return kite

def generate_login_url():
    """
    Generates the Zerodha login link.
    """
    kite = get_kite()
    return kite.login_url()

def exchange_request_token(request_token):
    """
    Exchanges request_token for access_token and saves it.
    """
    api_key, api_secret, _, _ = config.load_credentials()
    
    if not api_key or not api_secret:
        raise Exception("❌ API Key or Secret missing in config/config.json")

    kite = KiteConnect(api_key=api_key)
    
    try:
        print("🔄 Exchanging Request Token...")
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        # Save using the centralized config function
        config.save_access_token(access_token)
        print("✅ Access Token Saved Successfully!")
        return access_token
    except Exception as e:
        print(f"❌ Token Exchange Failed: {e}")
        return None
