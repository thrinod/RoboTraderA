import asyncio
import os
import sys
import json
from datetime import datetime
from app.services.upstox_service import upstox_service
from motor.motor_asyncio import AsyncIOMotorClient

# Setup DB
MONGODB_URL = "mongodb://localhost:27017"
DB_NAME = "robotrader"

async def test_fetch():
    result_data = {}
    try:
        print("Connecting to DB...")
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DB_NAME]
        
        print("Loading Token...")
        await upstox_service.load_token(db)
        
        instrument_key = "NSE_INDEX|Nifty 50"
        
        data = await upstox_service.get_intraday_candles(instrument_key, "1minute")
        
        if not data:
            result_data["error"] = "No Data Returned"
            
        candles = data.get('candles', [])
        result_data["count"] = len(candles)
        
        if candles:
            highs = [c['high'] for c in candles if c['high'] is not None]
            lows = [c['low'] for c in candles if c['low'] is not None]
            
            result_data["max_high"] = max(highs) if highs else 0
            result_data["min_low"] = min(lows) if lows else 0
            result_data["sample_first"] = candles[0]
            result_data["sample_last"] = candles[-1]
            
            # Outliers
            outliers = []
            for c in candles:
                if c['high'] > 50000 or c['high'] < 10000:
                    outliers.append(c)
            result_data["outliers"] = outliers

    except Exception as e:
        import traceback
        result_data["error"] = str(e)
        result_data["trace"] = traceback.format_exc()

    with open("backend/debug_results.json", "w") as f:
        json.dump(result_data, f, indent=2, default=str)
    print("Done writing to backend/debug_results.json")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_fetch())
