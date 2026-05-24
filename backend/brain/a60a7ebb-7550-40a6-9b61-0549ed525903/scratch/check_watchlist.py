import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_watchlist():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    print("Checking watchlist...")
    cursor = db["watchlist"].find({})
    async for doc in cursor:
        key = doc.get("instrument_key", "")
        if "BSE_FO" in key:
            print(f"FOUND BSE_FO in watchlist: {doc}")

if __name__ == "__main__":
    asyncio.run(check_watchlist())
