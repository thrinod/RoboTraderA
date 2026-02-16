import asyncio
import os
import sys
from app.services.upstox_service import UpstoxService
from motor.motor_asyncio import AsyncIOMotorClient

# Setup
os.environ["MONGODB_URL"] = "mongodb://localhost:27017" # Adjust if needed
service = UpstoxService()

async def test_logic():
    print("Testing UpstoxService Logic...")
    
    # Needs DB for token? 
    # Logic: load_token checks DB.
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["robotrader"]
    await service.load_token(db)
    
    if not service.access_token:
        print("WARNING: No access token loaded. API calls might fail if not using history cache/mock?")
        # Actually Upstox needs token for history API usually.
    
    key = "NSE_EQ|INE0LLY01014" # From previous run
    interval = "day"

    print(f"Fetching data for {key} interval {interval}...")
    try:
        data = await service.get_intraday_candles(key, interval)
        if data:
            print("Data Keys:", data.keys())
            print("Indicators:", data.get('indicators'))
        else:
            print("No data returned.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Redirect for capture
    sys.stdout = open('test_service_output.txt', 'w')
    sys.stderr = sys.stdout
    asyncio.run(test_logic())
    sys.stdout.close()
