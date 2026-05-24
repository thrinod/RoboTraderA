import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_deployments():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    count = await db["strategy_deployments"].count_documents({})
    print(f"Total strategy_deployments: {count}")
    
    cursor = db["strategy_deployments"].find({})
    async for doc in cursor:
        print(f"Deployment: {doc.get('primary_instrument')} ({doc.get('status')})")
        if "BSE_FO" in str(doc):
            print(f" !!! FOUND BSE_FO in this doc !!!")

if __name__ == "__main__":
    asyncio.run(check_deployments())
