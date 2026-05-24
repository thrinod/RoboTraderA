import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def stop_deployment():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    db = client[os.getenv("DB_NAME", "robotrader")]
    
    res = await db["strategy_deployments"].update_many(
        {"primary_instrument": "BSE_FO|1110534", "status": "ACTIVE"},
        {"$set": {"status": "STOPPED", "stopped_at": "2026-05-15T15:15:00", "error_reason": "Instrument expired/Invalid for Historical API"}}
    )
    print(f"Stopped {res.modified_count} deployments for BSE_FO|1110534")

if __name__ == "__main__":
    asyncio.run(stop_deployment())
