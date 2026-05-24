import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_deprecated():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    collections = ["scanner_instruments", "scanner_latest_results", "scanner_instruments_main", "scanner_results"]
    
    for coll in collections:
        print(f"Checking {coll}...")
        doc = await db[coll].find_one({"instrument_key": {"$regex": "BSE_FO"}})
        if doc:
            print(f" -> FOUND BSE_FO in {coll}: {doc}")
        else:
            print(f" -> No BSE_FO in {coll}")

if __name__ == "__main__":
    asyncio.run(check_deprecated())
