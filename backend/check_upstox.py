import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def run():
    client = AsyncIOMotorClient('mongodb://localhost:27017/')
    db = client['robotrader']
    docs = await db.upstox_collection.find({'trading_symbol': {'$in': ['TCS', 'TCS-EQ', 'RELIANCE', 'RELIANCE-EQ']}}).to_list(10)
    print([d.get('instrument_type') for d in docs])
    print([d.get('trading_symbol') for d in docs])
    print(docs[0] if docs else 'No docs')

asyncio.run(run())
