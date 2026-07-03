import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import sys
import os
import pandas as pd
import datetime

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

async def main():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["robotrader"]
    
    from app.services.upstox_service import upstox_service
    await upstox_service.load_token(db)
    
    # Fetch a sample instrument key from scanner_results
    sample = await db["scanner_results"].find_one({})
    if not sample:
        print("No scanner results found in DB!")
        return
        
    key = sample["instrument_key"]
    print(f"Sample stock: {key}")
    
    # Simulate process_instrument_core logic
    daily_df = await upstox_service._fetch_historical_df(key, "day", days_back_override=375)
    if daily_df is None or daily_df.empty:
        print("No daily history found!")
        return
        
    daily_closes = daily_df['close'].dropna()
    closes_list = [float(c) for c in daily_closes]
    
    # Sample real_ltp (suppose today's live price is 5% higher than yesterday close to make today's change very obvious!)
    yesterday_close = float(daily_closes.iloc[-1])
    real_ltp = yesterday_close * 1.05  # +5% today
    
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    last_date = daily_closes.index[-1].strftime('%Y-%m-%d')
    
    print(f"Today's date: {today_str}")
    print(f"Last daily candle date: {last_date}")
    print(f"Yesterday's close: {yesterday_close}")
    print(f"Simulated live LTP (today): {real_ltp} (+5.00%)")
    
    # Unified closes_list construction
    test_closes = list(closes_list)
    if last_date != today_str:
        test_closes.append(float(real_ltp))
        
    m = len(test_closes)
    print(f"Closes list length: {m}")
    print(f"Last 3 closes: {test_closes[-3:]}")
    
    # Calculate 7D close
    close_7d = round(test_closes[-8], 2)
    change_7d = round(((test_closes[-1] - close_7d) / close_7d) * 100, 2)
    print(f"\nCalculated 7D close (8th from end): {close_7d}")
    print(f"Calculated 7D change: {change_7d}%")
    
    # Compare with closes_list excluding today
    close_7d_ex = round(closes_list[-7], 2) # 7th from end of daily_closes
    change_7d_ex = round(((yesterday_close - close_7d_ex) / close_7d_ex) * 100, 2)
    print(f"If we excluded today's change (yesterday vs 7 trading days ago): {change_7d_ex}%")
    
    # Check what was in the database sample record!
    print(f"\n--- Database Record Values ---")
    print(f"DB Record LTP: {sample.get('ltp')}")
    print(f"DB Record change_7d: {sample.get('change_7d')}")
    print(f"DB Record close_7d: {sample.get('close_7d')}")

if __name__ == "__main__":
    asyncio.run(main())
