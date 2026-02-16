import asyncio
import os
import sys
import time
import json

# Adjust path to allow imports from app
sys.path.append(os.getcwd())

from app.services.upstox_service import upstox_service
from motor.motor_asyncio import AsyncIOMotorClient

# Setup DB
MONGODB_URL = "mongodb://localhost:27017"
DB_NAME = "robotrader"

async def test_fetch():
    print("Connecting to DB...")
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    
    print("Loading Token...")
    try:
        await upstox_service.load_token(db)
    except Exception as e:
        print(f"Failed to load token: {e}")
        return
    
    if not upstox_service.access_token:
        print("WARNING: No Access Token loaded. Functionality might be limited (mock data or partial).")

    instrument_key = "NSE_INDEX|Nifty 50"
    
    print(f"Fetching data for {instrument_key}...")
    start_time = time.time()
    try:
        data = await upstox_service.get_intraday_candles(instrument_key, "1minute")
    except Exception as e:
        print(f"Fetch Error: {e}")
        return

    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Fetch completed in {duration:.4f} seconds")
    
    if duration > 8.0:
        print(f"FAILED: Duration {duration:.4f}s exceeded 8s limit")
    else:
        print(f"SUCCESS: Duration {duration:.4f}s is under 8s limit")

    candles = []
    if not data:
        print("ERROR: No Data Returned")
    else:
        candles = data.get('candles', [])
        print(f"Fetched {len(candles)} candles")
        if 'indicators' in data:
            print(f"Indicators present: {list(data['indicators'].keys())}")
        if 'pivots' in data: 
            pass
        if 'indicators' in data and 'pivot_points' in data['indicators']:
             print(f"Pivot Points: {data['indicators']['pivot_points']}")

    result = {
        "duration": duration,
        "success": duration <= 8.0,
        "candles_count": len(candles),
        "indicators": list(data.get('indicators', {}).keys()) if data else []
    }
    
    with open("verification_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Written to verification_results.json")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_fetch())
