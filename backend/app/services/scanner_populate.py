
import httpx
import pandas as pd
from urllib.parse import quote

# Static list fallback if API fails or for speed
NIFTY_50_STATIC = [
    "NSE_EQ|INE742F01042", "NSE_EQ|INE021A01026", "NSE_EQ|INE238A01034", "NSE_EQ|INE257A01026", "NSE_EQ|INE002A01018",
    "NSE_EQ|INE040A01034", "NSE_EQ|INE090A01021", "NSE_EQ|INE860A01027", "NSE_EQ|INE018A01030", "NSE_EQ|INE009A01021",
    "NSE_EQ|INE296A01024", "NSE_EQ|INE154A01025", "NSE_EQ|INE062A01020", "NSE_EQ|INE019A01038", "NSE_EQ|INE585B01010",
    "NSE_EQ|INE669E01016", "NSE_EQ|INE242A01010", "NSE_EQ|INE245A01021", "NSE_EQ|INE528G01035", "NSE_EQ|INE192A01025",
    "NSE_EQ|INE795G01014", "NSE_EQ|INE467B01029", "NSE_EQ|INE155A01022", "NSE_EQ|INE038A01020", "NSE_EQ|INE237A01028",
    "NSE_EQ|INE030A01027", "NSE_EQ|INE213A01029", "NSE_EQ|INE917I01010", "NSE_EQ|INE003A01024", "NSE_EQ|INE101A01026",
    "NSE_EQ|INE752E01010", "NSE_EQ|INE469A01026", "NSE_EQ|INE015A01025", "NSE_EQ|INE002A01018", "NSE_EQ|INE044A01036",
    "NSE_EQ|INE280A01028", "NSE_EQ|INE205A01025", "NSE_EQ|INE112A01023", "NSE_EQ|INE081A01020", "NSE_EQ|INE397D01024",
    "NSE_EQ|INE848E01016", "NSE_EQ|INE160A01022", "NSE_EQ|INE148I01020", "NSE_EQ|INE481G01011", "NSE_EQ|INE018E01016",
    "NSE_EQ|INE032A01047", "NSE_EQ|INE029A01011", "NSE_EQ|INE121J01017", "NSE_EQ|INE733E01010", "NSE_EQ|INE075A01022"
]
# Note: Static ISINs are risky as they change, Symbols are better but Upstox needs ISIN/Key often.
# Better strategy: Download CSV from NSE daily or use Upstox Instrument File.

class ScannerPopulateService:
    def __init__(self, db, upstox_service):
        self.db = db
        self.upstox = upstox_service
        # Upstox Instrument File URL (Gzip)
        self.instrument_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"

    async def populate_index(self, index_name: str):
        """
        Populate DB with instruments from a specific index.
        Currently supports: NIFTY 50, BANK NIFTY (via static lists or broad search if feasible)
        For now, we will use a robust approach: Fetch ALL NSE Equity from Upstox master list, 
        filter by top 50/Index if we had that data, but Upstox Master list doesn't have "Index" column.
        
        So we must rely on a known list of Symbols for the index.
        """
        stock_list = []
        
        if index_name == "NIFTY 50":
            # Using Symbols is safer than ISINs for static lists
            stock_list = [
                "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "INFY", "ITC", "SBIN", "LICI", "HINDUNILVR",
                "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", "ADANIENT", "TITAN", "TATAMOTORS", "ULTRACEMCO", "AXISBANK",
                "NTPC", "ONGC", "KAJARIACER", "POWERGRID", "ADANIPORTS", "M&M", "TATASTEEL", "COALINDIA", "BAJAJFINSV", "WIPRO",
                "NESTLEIND", "JSWSTEEL", "TECHM", "GRASIM", "ADANIGREEN", "ADANIPOWER", "HDFCLIFE", "SBILIFE", "DRREDDY", "CIPLA",
                "TATACONSUM", "BRITANNIA", "APOLLOHOSP", "EICHERMOT", "DIVISLAB", "BAJAJ-AUTO", "HINDALCO", "LTIM", "HEROMOTOCO"
            ]
        elif index_name == "NIFTY BANK":
            stock_list = [
                "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB", "IDFCFIRSTB", "AUBANK",
                "FEDERALBNK", "BANDHANBNK"
            ]
        
        if not stock_list:
            return {"status": "error", "message": "Index not supported yet"}

        # 1. Fetch NSE Master List to get Instrument Keys
        # We can use the 'search_instruments' function in loop (slow) or fetch master list (heavy).
        # Let's use search_instruments loop for 50 items? It might take 10s. Acceptable for a manual trigger.
        
        added_count = 0
        from pymongo import UpdateOne
        operations = []
        import datetime

        for symbol in stock_list:
            # Search specifically in NSE_EQ
            try:
                # Mocking a search function here, assuming we can access it or use upstox service
                # We need to use the actual search API from upstox_service
                pass 
            except:
                pass

        # Actually, let's implement a smarter way. 
        # Since we don't have the master list loaded in memory, we will call search API.
        # But to avoid 50 HTTP calls in serial, we can do them in parallel with limit.
        
        import asyncio
        semaphore = asyncio.Semaphore(5) # 5 concurrent searches
        
        async def fetch_and_prepare(sym):
            async with semaphore:
                try:
                    # Search for "symbol" in NSE_EQ
                    res = await self.upstox.search_instruments(sym, segment="NSE_EQ")
                    # Find exact match
                    match = None
                    if res:
                        for item in res:
                            if item['trading_symbol'] == sym and item['exchange'] == 'NSE':
                                match = item
                                break
                        # If no exact match, take first
                        if not match and res:
                            match = res[0]
                    
                    if match:
                        return {
                            "instrument_key": match['instrument_key'],
                            "name": match['name'],
                            "exchange": match['exchange'],
                            "segment": match.get('segment', 'NSE_EQ'),
                            "trading_symbol": match['trading_symbol'],
                            "mtf_enabled": False, # Default
                            "added_at": datetime.datetime.now().isoformat()
                        }
                except Exception as e:
                    print(f"Error searching {sym}: {e}")
                return None

        tasks = [fetch_and_prepare(sym) for sym in stock_list]
        results = await asyncio.gather(*tasks)
        
        clean_results = [r for r in results if r]
        
        for item in clean_results:
            operations.append(
                UpdateOne(
                    {"instrument_key": item["instrument_key"]},
                    {"$set": item},
                    upsert=True
                )
            )

        if operations:
            await self.db["scanner_instruments"].bulk_write(operations)
            return {"status": "success", "message": f"Added {len(operations)} instruments to Scanner"}
        
        return {"status": "warning", "message": "No instruments found to add"}

