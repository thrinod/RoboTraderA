
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
        import datetime
        
        # Clear existing for "Load Only" behavior
        await self.db["scanner_instruments_main"].delete_many({})

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
            await self.db["scanner_instruments_main"].bulk_write(operations)
            return {"status": "success", "message": f"Added {len(operations)} instruments to Scanner Main"}
        
        return {"status": "warning", "message": "No instruments found to add"}


    async def populate_from_fno(self):
        """
        Populate scanner with stocks present in 'fno_stocks' collection.
        We match them against 'upstox_collection' (NSE_EQ) to get instrument keys.
        """
        try:
            # 1. Fetch FNO Symbols
            fno_cursor = self.db["fno_stocks"].find({})
            fno_list = await fno_cursor.to_list(length=1000)
            
            fno_symbols = set()
            for doc in fno_list:
                # Handle messy keys like "SYMBOL \n"
                for k, v in doc.items():
                    if "SYMBOL" in k.upper():
                        if isinstance(v, str):
                            clean_sym = v.strip().upper()
                            fno_symbols.add(clean_sym)
                        break
            
            if not fno_symbols:
                return {"status": "error", "message": "No symbols found in fno_stocks"}

            print(f"Found {len(fno_symbols)} FNO Symbols. Matching with Upstox DB...")

            # 2. Match with Upstox Collection (NSE_EQ)
            # Auto-fetch master instruments if collection is empty
            eq_count = await self.db["upstox_collection"].count_documents({"exchange": {"$in": ["NSE_EQ", "NSE"]}})
            if eq_count == 0:
                print("upstox_collection is empty! Auto-fetching master instruments...")
                await self.fetch_master_instruments()

            # Fetch all possible NSE Equity instruments
            # We fetch more fields to help with matching
            all_eq_cursor = self.db["upstox_collection"].find(
                {"exchange": {"$in": ["NSE_EQ", "NSE"]}},
                {"trading_symbol": 1, "instrument_key": 1, "name": 1, "instrument_type": 1, "exchange": 1}
            )
            all_eq_docs = await all_eq_cursor.to_list(length=50000)
            
            print(f"Loaded {len(all_eq_docs)} NSE instruments from DB for matching.")
            
            # Create a lookup map for faster matching
            # Format: { CLEAN_SYMBOL: DOC }
            upstox_map = {}
            for doc in all_eq_docs:
                ts = doc.get('trading_symbol', '').upper().strip()
                itype = doc.get('instrument_type', '').upper()
                
                # Only interested in Equities for FNO matching
                if itype not in ['EQ', 'EQUITY']:
                    continue
                
                # Normalize symbol: RELIANCE-EQ -> RELIANCE
                clean_ts = ts.replace("-EQ", "").strip()
                
                # If there are duplicates, prioritize NSE_EQ exchange
                if clean_ts not in upstox_map or doc.get('exchange') == 'NSE_EQ':
                    upstox_map[clean_ts] = doc

            matches = []
            matched_symbols = set()
            
            # Match FNO symbols against our map
            for sym in fno_symbols:
                if sym in upstox_map:
                    matches.append(upstox_map[sym])
                    matched_symbols.add(sym)
            
            print(f"Matched {len(matches)} instruments out of {len(fno_symbols)} FNO symbols.")

            # Index symbol -> Upstox instrument key mapping
            INDEX_KEY_MAP = {
                "NIFTY": "NSE_INDEX|Nifty 50",
                "BANKNIFTY": "NSE_INDEX|Nifty Bank",
                "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
                "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
                "SENSEX": "BSE_INDEX|SENSEX",
                "BANKEX": "BSE_INDEX|BANKEX",
                "NIFTYNXT50": "NSE_INDEX|Nifty Next 50",
                "NIFTYIT": "NSE_INDEX|Nifty IT",
            }

            # Check missing symbols
            still_missing = sorted(list(fno_symbols - matched_symbols))
            index_matches = []
            final_missing = []

            for sym in still_missing:
                if sym in INDEX_KEY_MAP:
                    index_matches.append({
                        "instrument_key": INDEX_KEY_MAP[sym],
                        "trading_symbol": sym,
                        "name": sym,
                        "instrument_type": "INDEX"
                    })
                    matched_symbols.add(sym)
                else:
                    final_missing.append(sym)

            if still_missing:
                print(f"=== TRULY MISSING FNO INSTRUMENTS ({len(final_missing)}) ===")
                for sym in final_missing:
                    print(f"  MISSING FNO: {sym}")
                print(f"=== END MISSING FNO ===")
            
            if len(matches) == 0 and len(index_matches) == 0:
                 return {
                     "status": "warning",
                     "message": "No matches found. Check if master instruments are loaded.",
                     "count": 0,
                     "debug": {
                         "db_total_nse": len(all_eq_docs),
                         "upstox_map_size": len(upstox_map),
                         "sample_fno": list(fno_symbols)[:5],
                         "sample_upstox": list(upstox_map.keys())[:5]
                     }
                 }

            # 3. Populate Scanner
            from pymongo import UpdateOne
            import datetime
            operations = []
            
            # Clear existing? Or just add?
            # User request: "load only these instrument". So we should Clear first.
            await self.db["scanner_instruments_main"].delete_many({})
            
            timestamp = datetime.datetime.now().isoformat()
            
            # Add equity matches
            for m in matches:
                doc = {
                    "instrument_key": m.get("instrument_key"),
                    "trading_symbol": m.get("trading_symbol"),
                    "name": m.get("name"),
                    "added_at": timestamp
                }
                
                operations.append(
                    UpdateOne(
                        {"instrument_key": m.get("instrument_key")},
                        {"$set": doc},
                        upsert=True
                    )
                )

            # Add index matches
            for idx in index_matches:
                doc = {
                    "instrument_key": idx["instrument_key"],
                    "trading_symbol": idx["trading_symbol"],
                    "name": idx["name"],
                    "added_at": timestamp
                }
                operations.append(
                    UpdateOne(
                        {"instrument_key": idx["instrument_key"]},
                        {"$set": doc},
                        upsert=True
                    )
                )

            if operations:
                await self.db["scanner_instruments_main"].bulk_write(operations)
            
            total_count = len(matches) + len(index_matches)
            return {
                "status": "success", 
                "message": f"Populated {total_count} FNO instruments ({len(matches)} stocks + {len(index_matches)} indices)",
                "count": total_count,
                "missing": len(still_missing),
                "missing_symbols": still_missing
            }

        except Exception as e:
            print(f"Error populating FNO: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def fetch_fno_list(self):
        """
        Downloads the official FNO list (Market Lots) from NSE website
        and updates the 'fno_stocks' collection.
        URL: https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv
        """
        try:
            url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            print(f"Downloading FNO list from {url}...")
            async with httpx.AsyncClient(verify=False) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    return {"status": "error", "message": f"Failed to download FNO list. Status: {resp.status_code}"}
                
                content = resp.content
            
            # Parse CSV
            import io
            df = pd.read_csv(io.BytesIO(content))
            
            # Clean column names (strip spaces)
            df.columns = [c.strip() for c in df.columns]
            
            # Filter for Equity Derivatives (usually first column 'UNDERLYING' or 'SYMBOL')
            # The CSV usually has columns like: UNDERLYING, SYMBOL, NOV-23, DEC-23, JAN-24...
            # We just need the SYMBOL column.
            
            possible_cols = ['SYMBOL', 'Symbol', 'UNDERLYING']
            symbol_col = next((c for c in possible_cols if c in df.columns), None)
            
            if not symbol_col:
                return {"status": "error", "message": f"Could not find Symbol column in CSV. Cols: {df.columns}"}
            
            # Extract unique symbols
            # Remove "NIFTY", "BANKNIFTY" etc if we only want stocks? 
            # User said "fno supported stock". Usually indices are also in this list.
            # Let's keep everything for now, scanning logic handles matching.
            
            symbols = df[symbol_col].dropna().unique().tolist()
            symbols = [s.strip().upper() for s in symbols if isinstance(s, str)]
            
            # Filter out header garbage if any
            symbols = [s for s in symbols if len(s) > 1 and "UNDERLYING" not in s]
            
            print(f"Extracted {len(symbols)} symbols from NSE CSV.")
            
            # Update DB
            if not symbols:
                return {"status": "error", "message": "No symbols found in CSV"}
            
            # Clear existing FNO collection
            await self.db["fno_stocks"].delete_many({})
            
            # Insert new
            from pymongo import InsertOne
            ops = [InsertOne({"SYMBOL": s}) for s in symbols]
            
            if ops:
                await self.db["fno_stocks"].bulk_write(ops)
            
            return {
                "status": "success", 
                "message": f"Updated FNO list with {len(symbols)} symbols from NSE",
                "count": len(symbols)
            }
            
        except Exception as e:
            print(f"Error fetching FNO list: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def fetch_master_instruments(self):
        """
        Downloads and parses the Upstox Master Instrument file (NSE.csv.gz).
        Populates 'upstox_collection' with ALL NSE instruments.
        URL: https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz
        """
        try:
            url = self.instrument_url
            print(f"Downloading Master Instruments from {url}...")
            
            async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return {"status": "error", "message": f"Failed to download Master list. Status: {resp.status_code}"}
                
                content = resp.content

            print(f"Downloaded {len(content)} bytes. Decompressing...")
            
            import gzip
            import io
            
            # Decompress
            with gzip.open(io.BytesIO(content), 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
            
            print(f"Loaded DataFrame with {len(df)} rows. Columns: {df.columns.tolist()}")
            
            # Normalize columns if needed. Upstox CSV usually has:
            # instrument_key, exchange_token, tradingsymbol, name, last_price, expiry, strike, tick_size, lot_size, instrument_type, option_type, exchange
            
            # We want to store everything in upstox_collection, but filtered for active?
            # Let's clean column names
            df.columns = [c.strip() for c in df.columns]
            
            # Normalize column names to match our schema
            # CSV has 'tradingsymbol' but our code expects 'trading_symbol'
            if 'tradingsymbol' in df.columns and 'trading_symbol' not in df.columns:
                df.rename(columns={'tradingsymbol': 'trading_symbol'}, inplace=True)
                print("Renamed column 'tradingsymbol' -> 'trading_symbol'")
            
            # Filter for NSE_EQ and NSE_FO? 
            # The file is "NSE.csv.gz" so it contains NSE_EQ and NSE_FO usually.
            # We need to ensure we map fields correctly for our schema.
            # Schema needs: instrument_key, trading_symbol, name, exchange, instrument_type
            
            if 'instrument_key' not in df.columns:
                 return {"status": "error", "message": "Invalid CSV format: Missing instrument_key"}

            # Prepare for Bulk Insert
            from pymongo import InsertOne, DeleteMany
            
            # Convert to list of dicts
            records = df.to_dict('records')
            
            # Batch Insert
            batch_size = 5000
            total_inserted = 0
            
            # Clear existing? Yes, full refresh.
            await self.db["upstox_collection"].delete_many({})
            print("Cleared existing upstox_collection.")
            
            operations = []
            for row in records:
                # Clean row data (handle NaNs)
                clean_row = {k: v for k, v in row.items() if pd.notna(v)}
                
                # Ensure validation fields
                if 'exchange' not in clean_row: clean_row['exchange'] = 'NSE' # Default
                
                # Helper for matching
                itype = clean_row.get('instrument_type')
                exchange = clean_row.get('exchange')
                
                # Check for Equity types
                is_equity = itype in ['EQUITY', 'EQ']
                
                if is_equity:
                      clean_row['instrument_type'] = 'EQ' # Normalize
                      if exchange == 'NSE':
                          clean_row['exchange'] = 'NSE_EQ'
                      
                      # Use the instrument_key exactly as provided in the Upstox CSV
                      # (which for NSE_EQ is usually ISIN-based, e.g. NSE_EQ|INE...)
                      
                      # Debug Log for crucial tickers
                      if clean_row.get('trading_symbol') in ['KOTAKBANK', 'MCX', 'NUVAMA']:
                          print(f"DEBUG: Ticker {clean_row['trading_symbol']} -> Key: {clean_row.get('instrument_key')}")
                
                if clean_row.get('trading_symbol') in ['KOTAKBANK', 'MCX', 'NUVAMA']:
                     try:
                         with open("ingestion_debug.txt", "a") as f:
                             f.write(f"RAW: {clean_row}\n")
                     except: pass
                
                # Format Expiry for Option Chain
                expiry = clean_row.get('expiry')
                if expiry:
                    # If it's a timestamp (int/float), convert to YYYY-MM-DD?
                    # Upstox CSV usually has DD-MMM-YYYY or YYYY-MM-DD.
                    # If it's already a string, keep it.
                    # If it's empty, ignore.
                    clean_row['expiry'] = str(expiry).strip() 
                    
                    # Optional: Convert DD-MM-YYYY to YYYY-MM-DD if needed, but string is likely fine for filtering.
                    # We can try to parse it with pandas if we want uniformity.
                    try:
                        dt = pd.to_datetime(expiry, errors='ignore') 
                        if not pd.isna(dt) and hasattr(dt, 'strftime'):
                             clean_row['expiry'] = dt.strftime('%Y-%m-%d')
                    except:
                        pass
                
                operations.append(InsertOne(clean_row))
                
                if len(operations) >= batch_size:
                    await self.db["upstox_collection"].bulk_write(operations)
                    total_inserted += len(operations)
                    operations = []
                    print(f"Inserted {total_inserted} instruments...")
            
            if operations:
                await self.db["upstox_collection"].bulk_write(operations)
                total_inserted += len(operations)
            
            return {
                "status": "success", 
                "message": f"Successfully loaded {total_inserted} instruments from Upstox Master List",
                "count": total_inserted
            }

        except Exception as e:
            print(f"Error fetching Master Instruments: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
