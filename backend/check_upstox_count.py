import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def run():
    client = AsyncIOMotorClient('mongodb://localhost:27017/')
    db = client['robotrader']
    c1 = await db.upstox_collection.count_documents({'exchange': {'$in': ['NSE_EQ', 'NSE']}})
    c2 = await db.upstox_collection.count_documents({})
    print('Filtered:', c1, 'Total:', c2)

asyncio.run(run())
