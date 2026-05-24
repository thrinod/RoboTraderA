import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_deployment_details():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    cursor = db["strategy_deployments"].find({"primary_instrument": "BSE_FO|1110534"})
    async for doc in cursor:
        print(f"Deployment Details: {doc}")

if __name__ == "__main__":
    asyncio.run(check_deployment_details())
