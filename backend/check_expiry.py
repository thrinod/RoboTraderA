
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

async def main():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client["robotrader"]
    
    print("\n--- Checking Expiry in upstox_collection ---")
    # Find a document with expiry
    doc = await db["upstox_collection"].find_one({"expiry": {"$exists": True}})
    
    if doc:
        print(f"Found Doc with Expiry: {doc.get('trading_symbol')}")
        print(f"Expiry: {doc.get('expiry')} (Type: {type(doc.get('expiry'))})")
    else:
        print("No document with 'expiry' field found.")
        
    # Check for NSE_FO specifically
    doc_fo = await db["upstox_collection"].find_one({"exchange": "NSE_FO"})
    if doc_fo:
         print(f"NSE_FO Doc: {doc_fo.get('trading_symbol')}")
         print(f"Expiry: {doc_fo.get('expiry')}")
         print(f"Keys: {list(doc_fo.keys())}")

    client.close()

if __name__ == "__main__":
    asyncio.run(main())
