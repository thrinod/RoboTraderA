import httpx
import asyncio

BASE_URL = "http://localhost:8000"

async def test_key(name, key):
    print(f"--- Testing {name} ({key}) ---")
    async with httpx.AsyncClient() as client:
        # 1. Get Expiry
        resp = await client.get(f"{BASE_URL}/market/options/expiry?instrument_key={key}")
        data = resp.json()
        print(f"Expiry Status: {resp.status_code}")
        dates = data.get("data", [])
        print(f"Dates: {dates[:2]}")
        
        if not dates:
            print("No dates found. Key might be invalid for Option Contract API.\n")
            return

        date = dates[0]
        
        # 2. Get Chain & Spot
        # We also want to check Spot Price specifically, but our API wraps it.
        resp = await client.get(f"{BASE_URL}/market/options/chain?instrument_key={key}&expiry_date={date}")
        chain_data = resp.json()
        print(f"Chain Status: {resp.status_code}")
        spot = chain_data.get("spot_price")
        chain_len = len(chain_data.get("data", []))
        print(f"Spot Price: {spot}")
        print(f"Chain Len: {chain_len}")
        print("\n")

async def main():
    keys = [
        ("Midcap Select (Std)", "NSE_INDEX|Nifty Midcap Select"),
        ("Midcap (Symbol)", "NSE_INDEX|MIDCPNIFTY"),
        ("Midcap (Select 2)", "NSE_INDEX|NIFTY MID SELECT"),
        ("FinNifty", "NSE_INDEX|Nifty Fin Service"),
        ("FinNifty (Symbol)", "NSE_INDEX|FINNIFTY"),
        ("Bankex", "BSE_INDEX|BANKEX"), # Control
    ]
    
    for name, key in keys:
        await test_key(name, key)

if __name__ == "__main__":
    asyncio.run(main())
