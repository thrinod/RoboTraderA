import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_all_instr():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    print("Checking scanner_instruments_main...")
    cursor = db["scanner_instruments_main"].find({})
    async for doc in cursor:
        key = doc.get("instrument_key", "")
        if "BSE_FO" in key or "1110534" in key:
            print(f"FOUND problematic key: {doc}")
        elif not key.startswith("NSE_EQ") and not key.startswith("NSE_INDEX"):
            print(f"Non-NSE key found: {key} ({doc.get('trading_symbol')})")

    print("\nChecking scanner_results...")
    cursor = db["scanner_results"].find({})
    async for doc in cursor:
        key = doc.get("instrument_key", "")
        if "BSE_FO" in key or "1110534" in key:
            print(f"FOUND problematic result: {key}")

if __name__ == "__main__":
    asyncio.run(check_all_instr())
