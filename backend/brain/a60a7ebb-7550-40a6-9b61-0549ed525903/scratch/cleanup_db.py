
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def cleanup():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['robotrader']
    upstox = db['upstox_collection']
    
    # Count before
    count_before = await upstox.count_documents({})
    print(f"Total documents in upstox_collection: {count_before}")
    
    # Find documents to delete
    query = {"instrument_key": {"$exists": False}}
    to_delete = await upstox.count_documents(query)
    print(f"Documents to delete (missing instrument_key): {to_delete}")
    
    if to_delete > 0:
        res = await upstox.delete_many(query)
        print(f"Deleted {res.deleted_count} documents.")
    
    # Also check for null instrument_key
    query_null = {"instrument_key": None}
    to_delete_null = await upstox.count_documents(query_null)
    print(f"Documents to delete (null instrument_key): {to_delete_null}")
    
    if to_delete_null > 0:
        res = await upstox.delete_many(query_null)
        print(f"Deleted {res.deleted_count} documents.")

    count_after = await upstox.count_documents({})
    print(f"Total documents after cleanup: {count_after}")

if __name__ == "__main__":
    asyncio.run(cleanup())
