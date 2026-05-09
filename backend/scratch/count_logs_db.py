import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

async def count_logs():
    load_dotenv()
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DB_NAME = os.getenv("MONGODB_DB_NAME", "robotrader")
    
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    
    ids = ["69fddf77ab0e4234d9270631", "69fde57946807337aa6b8dad"]
    for dep_id in ids:
        count = await db["deployment_logs"].count_documents({"deployment_id": dep_id})
        print(f"Total logs for {dep_id}: {count}")

if __name__ == "__main__":
    asyncio.run(count_logs())
