import asyncio
from app.services.upstox_service import upstox_service
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

async def test_fetch():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    await upstox_service.load_token(db)
    print(f"Token loaded: {upstox_service.access_token[:10]}...")
    
    key = "NSE_EQ|INE100A01010" # ATUL
    interval = "1minute"
    
    print(f"Fetching history for {key} ({interval})...")
    df = await upstox_service._fetch_historical_df(key, interval)
    
    if df is not None:
        print(f"Fetched {len(df)} candles.")
        if not df.empty:
            print(f"First candle: {df.index[0]}")
            print(f"Last candle: {df.index[-1]}")
    else:
        print("Fetch returned None")

if __name__ == "__main__":
    asyncio.run(test_fetch())
