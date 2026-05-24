import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    sample = await db["instrument_candles"].find_one({})
    if sample:
        print(f"Sample instrument in DB: {sample.get('instrument_key')} | Interval: {sample.get('interval')}")
    
    keys = ["NSE_EQ|INE285H01022", "NSE_EQ|INE446A01025", "NSE_EQ|INE951D01028", "NSE_EQ|INE100A01010"]
    for key in keys:
        doc = await db["upstox_collection"].find_one({"instrument_key": key})
        print(f"Key: {key} -> {doc.get('name') if doc else 'NOT FOUND'} ({doc.get('trading_symbol') if doc else 'N/A'})")
        
        # Check if candles exist for 'day' interval
        candle_doc = await db["instrument_candles"].find_one({"instrument_key": key, "interval": "day"})
        print(f"  Candles (day) in DB: {len(candle_doc.get('candles', [])) if candle_doc else 0}")

if __name__ == "__main__":
    asyncio.run(check())
