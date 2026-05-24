import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_deployments():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    print("Checking strategy_deployments...")
    cursor = db["strategy_deployments"].find({})
    async for doc in cursor:
        key = doc.get("primary_instrument", "")
        if "BSE_FO" in key:
            print(f"FOUND BSE_FO in strategy_deployments: {doc}")

if __name__ == "__main__":
    asyncio.run(check_deployments())
