import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

async def list_unique_ids():
    load_dotenv()
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DB_NAME = os.getenv("MONGODB_DB_NAME", "robotrader")
    
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    
    ids = await db["deployment_logs"].distinct("deployment_id")
    print(f"Unique deployment_ids in logs: {ids}")

if __name__ == "__main__":
    asyncio.run(list_unique_ids())
