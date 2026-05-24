
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['robotrader']
    
    print("--- scanner_instruments ---")
    coll = db['scanner_instruments']
    count = await coll.count_documents({})
    print(f"Total count: {count}")
    sample = await coll.find().to_list(length=5)
    for s in sample:
        print(f"  {s}")
        
    print("\n--- upstox_collection indexes ---")
    upstox = db['upstox_collection']
    indexes = await upstox.index_information()
    print(f"Indexes: {indexes}")

    print("\n--- scanner_instruments_main ---")
    main_coll = db['scanner_instruments_main']
    indexes = await main_coll.index_information()
    print(f"Indexes: {indexes}")

if __name__ == "__main__":
    asyncio.run(check())
