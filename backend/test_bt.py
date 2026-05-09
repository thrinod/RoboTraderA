import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from app.services.upstox_service import UpstoxService
from app.services.backtest_service import BacktestService

async def main():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["robotrader"]
    
    # Needs some config, let's just see if we can instantiate and run
    upstox = UpstoxService()
    # It might fail if UPSTOX_API_KEY is not set, but we just need to see if it throws 500
    # Actually _fetch_historical_df needs a valid token? 
    # Yes, upstox_service gets it from DB.
    await upstox.load_token(db)
    
    backtester = BacktestService(upstox)
    
    try:
        res = await backtester.run_strategy("BSE_FO|888771", "15minute", 30)
        print("Result:", type(res))
        print(res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("i:/RoboTrader/backend/.env")
    asyncio.run(main())
