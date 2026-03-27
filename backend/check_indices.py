import asyncio
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from app.services.upstox_service import upstox_service

load_dotenv()

async def test():
    db = AsyncIOMotorClient(os.getenv('MONGODB_URL', 'mongodb://localhost:27017'))['robotrader']
    await upstox_service.load_token(db)
    print('Token:', upstox_service.has_valid_token())
    
    keys = [
        'NSE_INDEX|Nifty 50', 
        'NSE_INDEX|Nifty Bank', 
        'NSE_INDEX|Nifty Fin Service', 
        'BSE_INDEX|SENSEX'
    ]
    
    # Test getting spot prices
    print("--- SPOT PRICES ---")
    prices = []
    for k in keys:
        p = await upstox_service.get_spot_price(k)
        print(f"{k}: {p}")
        prices.append(p)
        
    print("\n--- OPTION CHAIN TOTALS ---")
    for k in keys:
        dates = await upstox_service.get_expiry_dates(k)
        if dates:
            chain = await upstox_service.get_option_chain(k, dates[0])
            print(f"{k} ({dates[0]}): CE Total OI: {chain['totals']['ce']}, PE Total OI: {chain['totals']['pe']}, Spot: {chain.get('spot_price')}")
        else:
            print(f"{k}: No expiry dates found")

if __name__ == "__main__":
    asyncio.run(test())
