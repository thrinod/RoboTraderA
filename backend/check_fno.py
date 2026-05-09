import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def run():
    client = AsyncIOMotorClient('mongodb://localhost:27017/')
    db = client['robotrader']
    docs = await db.fno_stocks.find().to_list(10)
    print(docs)

asyncio.run(run())
