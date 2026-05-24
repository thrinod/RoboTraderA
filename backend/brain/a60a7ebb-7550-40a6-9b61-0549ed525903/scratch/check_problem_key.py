import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_key():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    # 1. Check in scanner_instruments_main
    instr = await db["scanner_instruments_main"].find_one({"instrument_key": "BSE_FO|1110534"})
    print(f"In scanner_instruments_main: {instr}")
    
    # 2. Check in upstox_collection
    upstox = await db["upstox_collection"].find_one({"instrument_key": "BSE_FO|1110534"})
    print(f"In upstox_collection: {upstox}")
    
    # 3. Check what else matches this symbol if found
    if upstox:
        sym = upstox.get("trading_symbol")
        print(f"Searching for other instruments with symbol {sym}...")
        cursor = db["upstox_collection"].find({"trading_symbol": sym})
        async for doc in cursor:
            print(f" - {doc.get('instrument_key')} ({doc.get('exchange')})")

if __name__ == "__main__":
    asyncio.run(check_key())
