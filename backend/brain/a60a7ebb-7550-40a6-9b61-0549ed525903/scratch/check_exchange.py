
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['robotrader']
    coll = db['scanner_instruments_main']
    
    bse_count = await coll.count_documents({"exchange": {"$regex": "^BSE"}})
    nse_count = await coll.count_documents({"exchange": {"$regex": "^NSE"}})
    
    print(f"BSE Count: {bse_count}")
    print(f"NSE Count: {nse_count}")
    
    # Sample BSE
    if bse_count > 0:
        bse_sample = await coll.find({"exchange": {"$regex": "^BSE"}}).to_list(length=5)
        print(f"BSE Sample: {bse_sample}")

if __name__ == "__main__":
    asyncio.run(check())
