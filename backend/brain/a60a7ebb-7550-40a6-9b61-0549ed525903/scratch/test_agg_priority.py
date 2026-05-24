
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import datetime

async def test_agg():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['robotrader']
    
    added_at = datetime.datetime.now().isoformat()
    
    pipeline = [
        {
            "$lookup": {
                "from": "upstox_collection",
                "localField": "SYMBOL",
                "foreignField": "trading_symbol",
                "as": "master_data"
            }
        },
        {"$unwind": "$master_data"},
        {
            "$match": {
                "master_data.exchange": {"$in": ["NSE_EQ", "NSE", "BSE_EQ", "BSE"]},
                "master_data.instrument_key": {"$ne": None}
            }
        },
        {
            "$addFields": {
                "priority": {
                    "$cond": [
                        {"$in": ["$master_data.exchange", ["NSE_EQ", "NSE"]]},
                        1,
                        2
                    ]
                }
            }
        },
        {"$sort": {"SYMBOL": 1, "priority": 1}},
        {
            "$group": {
                "_id": "$SYMBOL",
                "instrument_key": {"$first": "$master_data.instrument_key"},
                "name": {"$first": {"$ifNull": ["$master_data.name", "$name"]}},
                "exchange": {"$first": {"$ifNull": ["$master_data.exchange", "$exchange"]}},
                "segment": {"$first": {"$ifNull": ["$master_data.segment", "$segment"]}},
                "trading_symbol": {"$first": "$master_data.trading_symbol"},
                "mtf_enabled": {"$first": {"$literal": False}},
                "added_at": {"$first": {"$literal": added_at}}
            }
        },
        {
            "$group": {
                "_id": "$instrument_key",
                "name": {"$first": "$name"},
                "exchange": {"$first": "$exchange"},
                "segment": {"$first": "$segment"},
                "trading_symbol": {"$first": "$trading_symbol"},
                "mtf_enabled": {"$first": "$mtf_enabled"},
                "added_at": {"$first": "$added_at"}
            }
        },
        {
            "$project": {
                "_id": 0,
                "instrument_key": "$_id",
                "name": 1,
                "exchange": 1,
                "segment": 1,
                "trading_symbol": 1,
                "mtf_enabled": 1,
                "added_at": 1
            }
        },
        {
            "$out": "scanner_instruments_main_test"
        }
    ]
    
    print("Running prioritized aggregation test...")
    import time
    start = time.time()
    await db["scanner_instruments"].aggregate(pipeline).to_list(length=1)
    end = time.time()
    print(f"Aggregation took {end - start:.2f} seconds.")
    
    count = await db["scanner_instruments_main_test"].count_documents({})
    bse_count = await db["scanner_instruments_main_test"].count_documents({"exchange": {"$regex": "^BSE"}})
    nse_count = await db["scanner_instruments_main_test"].count_documents({"exchange": {"$regex": "^NSE"}})
    
    print(f"Total count: {count}")
    print(f"NSE count: {nse_count}")
    print(f"BSE count: {bse_count}")

if __name__ == "__main__":
    asyncio.run(test_agg())
