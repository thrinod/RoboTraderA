
import sys
import os

# Ensure backend dir is in path
sys.path.append("i:/RoboTrader/backend")

try:
    from app.services.alice_blue_service import alice_blue_service
    print("AliceBlueService Import: SUCCESS")
    print(f"Service Type: {type(alice_blue_service)}")
    
    # Check internals
    print(f"Has Market Data (AliceBlue): {alice_blue_service.ab_market_data is not None}")
    print(f"Has Trading (TradeHub): {alice_blue_service.ab_trading is not None}")
    
    # Check Methods
    try:
         alice_blue_service.login("TEST", "TEST_KEY")
         print("Login Method Call: SUCCESS (Mock)")
    except Exception as e:
         print(f"Login Method Call Failed: {e}")

except ImportError as e:
    print(f"AliceBlueService Import FAILED: {e}")
except Exception as e:
    print(f"General Error: {e}")
