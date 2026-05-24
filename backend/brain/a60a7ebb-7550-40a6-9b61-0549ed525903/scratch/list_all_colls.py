import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def list_colls():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    colls = await db.list_collection_names()
    print(f"Collections in {os.getenv('DB_NAME', 'robotrader')}: {colls}")
    
    for coll in colls:
        count = await db[coll].count_documents({"instrument_key": {"$regex": "BSE_FO"}})
        if count > 0:
            print(f" -> {coll} has {count} BSE_FO entries")
            doc = await db[coll].find_one({"instrument_key": {"$regex": "BSE_FO"}})
            print(f"    Example: {doc}")

if __name__ == "__main__":
    asyncio.run(list_colls())
