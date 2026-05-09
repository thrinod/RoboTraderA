import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def run():
    client = AsyncIOMotorClient('mongodb://localhost:27017/')
    db = client['robotrader']
    
    fno_cursor = db.fno_stocks.find({})
    fno_list = await fno_cursor.to_list(length=1000)
    fno_symbols = set()
    for doc in fno_list:
        for k, v in doc.items():
            if "SYMBOL" in k.upper():
                if isinstance(v, str):
                    fno_symbols.add(v.strip().upper())
                break

    all_eq_cursor = db.upstox_collection.find(
        {"exchange": {"$in": ["NSE_EQ", "NSE"]}},
        {"trading_symbol": 1, "instrument_key": 1, "name": 1, "instrument_type": 1, "exchange": 1}
    )
    all_eq_docs = await all_eq_cursor.to_list(length=50000)
    
    upstox_map = {}
    for doc in all_eq_docs:
        ts = doc.get('trading_symbol', '').upper().strip()
        itype = doc.get('instrument_type', '')
        if itype:
            itype = itype.upper()
        else:
            itype = ''
        
        if itype not in ['EQ', 'EQUITY']:
            continue
        
        clean_ts = ts.replace("-EQ", "").strip()
        if clean_ts not in upstox_map or doc.get('exchange') == 'NSE_EQ':
            upstox_map[clean_ts] = doc

    matches = []
    matched_symbols = set()
    for sym in fno_symbols:
        if sym in upstox_map:
            matches.append(upstox_map[sym])
            matched_symbols.add(sym)

    still_missing = sorted(list(fno_symbols - matched_symbols))
    print(f"FNO count: {len(fno_symbols)}")
    print(f"Upstox map size: {len(upstox_map)}")
    print(f"Matches: {len(matches)}")
    print(f"Missing count: {len(still_missing)}")
    if still_missing:
        print("Missing sample:", still_missing[:10])

asyncio.run(run())
