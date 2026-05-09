import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

async def check_deployments():
    load_dotenv()
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DB_NAME = os.getenv("MONGODB_DB_NAME", "robotrader")
    
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    
    active_deployments = await db["strategy_deployments"].find().to_list(100)
    for dep in active_deployments:
        print(f"ID: {dep.get('_id')} - Status: {dep.get('status')} - Instrument: {dep.get('primary_instrument')}")

if __name__ == "__main__":
    asyncio.run(check_deployments())
