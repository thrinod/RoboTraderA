from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

# Import services
from app.services.upstox_service import upstox_service
from app.services.trading_engine import trading_engine
from app.services.mock_trade_service import mock_trade_service
from app.services.alice_blue_service import alice_blue_service
from app.services.alice_blue_service import alice_blue_service
from app.services.scanner_populate import ScannerPopulateService

# Global Service Instances
scanner_populate = None

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = "robotrader"

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Startup: Connect to MongoDB
        app.mongodb_client = AsyncIOMotorClient(MONGODB_URL)
        app.mongodb = app.mongodb_client[DB_NAME]
        
        # Load token from DB
        await upstox_service.load_token(app.mongodb)
        
        # Pass DB to Services
        trading_engine.set_db(app.mongodb)
        mock_trade_service.set_db(app.mongodb)
        
        # Initialize Scanner Populate Service
        global scanner_populate
        scanner_populate = ScannerPopulateService(app.mongodb, upstox_service)
        
        # Check if collection is empty and insert dummy data if needed
        count = await app.mongodb["upstox_collection"].count_documents({})
        if count == 0:
            # Insert market data sample
            await app.mongodb["upstox_collection"].insert_one({
                "ticker": "EXAMPLE_STOCK",
                "price": 123.45,
                "rsi": 65.5,
                "status": "Waiting for Live Data",
                "timestamp": "2024-01-01T12:00:00"
            })
            # Insert instrument data sample (Requested by user)
            await app.mongodb["upstox_collection"].insert_one({
                "weekly": False,
                "segment": "NCD_FO",
                "name": "JPYINR",
                "exchange": "NSE",
                "expiry": 1774636199000,
                "instrument_type": "CE",
                "asset_symbol": "JPYINR",
                "underlying_symbol": "JPYINR",
                "instrument_key": "NCD_FO|14294",
                "lot_size": 1,
                "freeze_quantity": 10000,
                "exchange_token": "14294",
                "minimum_lot": 1,
                "tick_size": 0.25,
                "asset_type": "CUR",
                "underlying_type": "CUR",
                "trading_symbol": "JPYINR 61 CE 27 MAR 26",
                "strike_price": 61,
                "qty_multiplier": 1000
            })
            print("Inserted dummy data into upstox_collection")
            
        print(f"Connected to MongoDB at {MONGODB_URL}")
        yield
    except Exception as e:
        print(f"CRITICAL STARTUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Shutdown: Close connection
        if hasattr(app, 'mongodb_client'):
            app.mongodb_client.close()
        print("Disconnected from MongoDB")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Hello from RoboTrader Backend!", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Upstox Auth Routes ---
@app.get("/auth/upstox")
async def login_upstox():
    return RedirectResponse(upstox_service.get_login_url())

@app.get("/auth/login_url")
async def get_login_url_json():
    url = upstox_service.get_login_url()
    return {"login_url": url}

@app.get("/auth/callback")
async def auth_callback(code: str):
    token = await upstox_service.exchange_code_for_token(code)
    return {"message": "Authenticated successfully", "access_token": token}

# --- Database Viewer Routes ---

def serialize_doc(doc):
    """Convert ObjectId to string for JSON serialization"""
    if doc.get("_id"):
        doc["_id"] = str(doc["_id"])
    return doc

@app.get("/db/collections")
async def list_collections():
    cols = await app.mongodb.list_collection_names()
    return {"collections": cols}


# --- Watchlist Routes ---

class WatchlistItem(BaseModel):
    instrument_key: str
    watchlist_id: int = 1 # Support 5 watchlists (1-5)
    added_at: str | None = None
    ltp: float | None = None
    change: float | None = None
    change_percent: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None

@app.get("/watchlist")
async def get_watchlist(watchlist_id: int = 1):
    cursor = app.mongodb["watchlist"].find({"watchlist_id": watchlist_id}).sort("added_at", -1)
    docs = await cursor.to_list(length=100)
    return {"data": [serialize_doc(doc) for doc in docs]}

class QuoteRequest(BaseModel):
    instrument_keys: list[str]

@app.post("/watchlist/quote")
async def get_quotes(req: QuoteRequest):
    # Fetch Market Data Only (No DB Save)
    if not req.instrument_keys:
        return {"data": {}}
    
    # Lazy load token if needed for manual fetch
    if not upstox_service.access_token:
        await upstox_service.load_token(app.mongodb)
        
    quotes = upstox_service.get_market_quotes(req.instrument_keys)
    
    # Filter response to only include requested keys to avoid duplication from internal aliasing
    # filtered_quotes = {}
    # for k in req.instrument_keys:
    #     if k in quotes:
    #         filtered_quotes[k] = quotes[k]
    
    # DEBUG: Return ALL quotes to see what we get
    # print(f"DEBUG: Watchlist Quote returning {len(quotes)} items for {req.instrument_keys}")
    return {"data": quotes}

@app.post("/watchlist/refresh")
async def refresh_watchlist(watchlist_id: int = 1):
    # Lazy Load Token if missing
    if not upstox_service.access_token:
        await upstox_service.load_token(app.mongodb)

    cursor = app.mongodb["watchlist"].find({"watchlist_id": watchlist_id}).sort("added_at", -1)
    docs = await cursor.to_list(length=100)
    
    keys = [doc["instrument_key"] for doc in docs if "instrument_key" in doc]
    if not keys:
         return {"data": []}

    # Fetch fresh data
    market_data = upstox_service.get_market_quotes(keys)
    
    # Bulk Update DB
    from pymongo import UpdateOne
    operations = []
    results = []
    
    import datetime
    now = datetime.datetime.now().isoformat()

    for doc in docs:
        key = doc.get("instrument_key")
        item = serialize_doc(doc)
        
        # Determine Name from upstox_collection if missing or always
        inst_doc = await app.mongodb["upstox_collection"].find_one({"instrument_key": key})
        if inst_doc:
            item["name"] = inst_doc.get("name")
        
        # Try finding the quote using key (Standard Match)
        quote = None
        if key and key in market_data:
             quote = market_data[key]
        
        if quote:
            # Update item locally
            item["ltp"] = quote.get("ltp")
            item["change"] = quote.get("change")
            item["change_percent"] = quote.get("change_percent")
            item["open"] = quote.get("open")
            item["high"] = quote.get("high")
            item["low"] = quote.get("low")
            item["updated_at"] = now
            
            # Prepare DB Update
            update_fields = {
                        "ltp": item["ltp"], 
                        "change": item["change"], 
                        "change_percent": item["change_percent"],
                        "open": item["open"],
                        "high": item["high"],
                        "low": item["low"],
                        "updated_at": now
            }
            if "name" in item:
                update_fields["name"] = item["name"]

            op = UpdateOne(
                    {"instrument_key": key},
                    {"$set": update_fields}
                )
            operations.append(op)
        elif "name" in item:
             # Persist name even if no quote update
             operations.append(UpdateOne(
                 {"instrument_key": key},
                 {"$set": {"name": item["name"]}}
             ))

        results.append(item)
    
    if operations:
        await app.mongodb["watchlist"].bulk_write(operations)
        
    return {"data": results}

@app.post("/watchlist")
async def add_to_watchlist(item: WatchlistItem):
    import datetime
    # Check if exists in this specific watchlist
    existing = await app.mongodb["watchlist"].find_one({
        "instrument_key": item.instrument_key,
        "watchlist_id": item.watchlist_id
    })
    if existing:
        return {"status": "error", "message": f"Already in Watchlist {item.watchlist_id}"}
    
    doc = item.dict()
    doc["added_at"] = datetime.datetime.now().isoformat()
    await app.mongodb["watchlist"].insert_one(doc)
    return {"status": "success", "message": f"Added to Watchlist {item.watchlist_id}"}

@app.delete("/watchlist/{instrument_key}")
async def remove_from_watchlist(instrument_key: str, watchlist_id: int = 1):
    res = await app.mongodb["watchlist"].delete_one({
        "instrument_key": instrument_key,
        "watchlist_id": watchlist_id
    })
    if res.deleted_count > 0:
        return {"message": "Removed from watchlist"}
    return {"message": "Item not found"}

@app.get("/db/data/{collection_name}")
async def get_collection_data(collection_name: str, search: str = None, limit: int = 100):
    query = {}
    if search:
        # Simple regex search across common string fields
        # Note: This is a basic implementation. For production, use text indexes.
        regex = {"$regex": search, "$options": "i"}
        query = {
            "$or": [
                {"ticker": regex},
                {"symbol": regex},
                {"name": regex},
                {"status": regex},
                {"instrument_key": regex}
            ]
        }
    
    cursor = app.mongodb[collection_name].find(query).sort("_id", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    return {"data": [serialize_doc(doc) for doc in docs]}

@app.get("/market/instruments/search")
async def search_instruments(q: str = "", limit: int = 50, segment: str = None, exchange: str = None, instrument_type: str = None, mtf_enabled: bool = False):
    # Allow search if query is at least 2 chars OR if any filter is applied
    has_filters = segment or exchange or instrument_type or mtf_enabled
    if (not q or len(q) < 2) and not has_filters:
        return {"data": []}
    
    regex = {"$regex": q, "$options": "i"}
    query_filters = []
    
    # Text Search - only apply if q is present
    if q and len(q) >= 2:
        query_filters.append({
            "$or": [
                {"name": regex},
                {"instrument_key": regex},
                {"trading_symbol": regex},
                {"isin": regex},
                 {"segment": regex},
                {"exchange": regex},
                {"instrument_type": regex}
            ]
        })

    # Apply Filters
    if segment:
        query_filters.append({"segment": segment})
    if exchange:
        query_filters.append({"exchange": exchange})
    if instrument_type:
        query_filters.append({"instrument_type": instrument_type})
    if mtf_enabled:
        # Assuming field name is mtf_enabled
        query_filters.append({"mtf_enabled": True})
    
    query = {"$and": query_filters} if query_filters else {}
    
    cursor = app.mongodb["upstox_collection"].find(query).limit(limit)
    docs = await cursor.to_list(length=limit)
    return {"data": [serialize_doc(doc) for doc in docs]}

@app.get("/market/instruments/types")
async def get_instrument_types():
    """Returns unique instrument types from upstox_collection"""
    types = await app.mongodb["upstox_collection"].distinct("instrument_type")
    # Filter out empty or None
    valid_types = [t for t in types if t]
    return {"data": sorted(valid_types)}

class TokenRequest(BaseModel):
    token: str

class OrderItem(BaseModel):
    instrument_key: str
    quantity: int
    transaction_type: str
    order_type: str = "MARKET"
    price: float = 0.0

class BulkOrderRequest(BaseModel):
    orders: List[OrderItem]

@app.post("/auth/token")
async def set_token(request: Request, token_req: TokenRequest):
    is_valid = upstox_service.verify_token(token_req.token)
    if is_valid:
        await upstox_service.save_token(request.app.mongodb, token_req.token)
        return {"message": "Token verified and connected", "status": "active", "upstox_status": "Connected"}
    else:
        return {"message": "Invalid Token", "status": "error", "upstox_status": "Disconnected"}

@app.get("/auth/status")
async def get_auth_status():
    if upstox_service.has_valid_token():
        return {"status": "authenticated", "upstox": "connected"}
    return {"status": "unauthenticated", "upstox": "disconnected"}

# --- Trading Routes ---
@app.get("/trade/start")
async def start_trading():
    trading_engine.start_trading()
    return {"status": "Trading Started"}

@app.get("/trade/stop")
async def stop_trading():
    trading_engine.stop_trading()
    return {"status": "Trading Stopped"}

@app.post("/trade/place_orders")
async def place_orders(req: BulkOrderRequest):
    results = []
    print(f"Received {len(req.orders)} orders")
    for order in req.orders:
        # Execute each order
        # Default quantity logic? For now assume frontend sends correct Lot Size or Qty
        res = upstox_service.place_order(
            instrument_key=order.instrument_key,
            quantity=order.quantity,
            transaction_type=order.transaction_type,
            order_type=order.order_type,
            price=order.price
        )
        results.append({"key": order.instrument_key, "result": res})
    
    return {"status": "completed", "results": results}

@app.post("/trade/cancel_all")
async def cancel_all_orders():
    result = await upstox_service.cancel_all_orders()
    return result

@app.post("/trade/square_off")
async def square_off_all():
    result = await upstox_service.square_off_all_positions()
    return result

class ExitRequest(BaseModel):
    instrument_key: str

@app.post("/trade/exit")
async def exit_position(req: ExitRequest):
    # Call async service method directly
    return await upstox_service.square_off_position(req.instrument_key)

@app.get("/user/funds")
def get_funds():
    data = upstox_service.get_funds()
    if not data:
        # Return empty structure if failed, so frontend doesn't crash but shows 0
        return {"data": {"equity": {"available_margin": 0.0, "used_margin": 0.0}}}
    return {"data": data} # Wrap in data key if upstox_service returns raw dict without 'data' wrapper (get_funds usually returns main obj)

@app.get("/user/positions")
def get_positions():
    data = upstox_service.get_positions()
    if not data:
        return {"data": []}
    return {"data": data}

@app.get("/user/holdings")
def get_holdings():
    data = upstox_service.get_holdings()
    if not data:
        return {"data": []}
    return data # get_holdings already wraps in {'data': ...} in service now

@app.get("/market/options/chain")
async def get_option_chain(instrument_key: str, expiry_date: str):
    # instrument_key example: NSE_INDEX|Nifty 50
    # expiry_date example: 2025-12-28
    spot_price = await upstox_service.get_spot_price(instrument_key)
    result = await upstox_service.get_option_chain(instrument_key, expiry_date)
    return {
        "data": result.get('chain', []),
        "spot_price": spot_price,
        "totals": result.get('totals', {'ce': 0, 'pe': 0})
    }

# --- Snapshots ---
class SnapshotItem(BaseModel):
    name: str = "" # e.g. "Index", "Call 1"
    instrument_key: str
    interval: str
    ltp: float = 0
    strike_price: float = 0
    indicators: dict = {}

class SnapshotRequest(BaseModel):
    snapshot_type: str # "BUY" or "SELL"
    notes: str = ""
    items: List[SnapshotItem]

@app.post("/analysis/snapshot")
async def save_snapshot(request: SnapshotRequest):
    import datetime
    
    doc = request.dict()
    doc["timestamp"] = datetime.datetime.now()
    
    # Store in "market_snapshots"
    res = await app.mongodb["market_snapshots"].insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    
    return {"status": "success", "id": doc["_id"]}

# --- Baskets ---
class BasketData(BaseModel):
    name: str = "Custom Basket"
    indexKey: str
    slot1Key: str
    slot1Label: str = ""
    slot1Strike: str = ""
    slot2Key: str
    slot2Label: str = ""
    slot2Strike: str = ""
    slot3Key: str
    slot3Label: str = ""
    slot3Strike: str = ""
    slot4Key: str
    slot4Label: str = ""
    slot4Strike: str = ""
    indexInterval: str
    slot1Interval: str
    slot2Interval: str
    slot3Interval: str
    slot4Interval: str

@app.post("/analysis/basket/{basket_id}")
async def save_basket(basket_id: int, data: BasketData):
    # Upsert basket
    await app.mongodb["analysis_baskets"].update_one(
        {"basket_id": basket_id},
        {"$set": data.dict()},
        upsert=True
    )
    return {"status": "success", "basket_id": basket_id}

@app.get("/analysis/basket/{basket_id}")
async def get_basket(basket_id: int):
    doc = await app.mongodb["analysis_baskets"].find_one({"basket_id": basket_id})
    if doc:
        doc.pop("_id", None)
        return doc
    return {} # Empty if not found

# Existing routes...
@app.get("/market/nifty/data")
async def get_nifty_data(interval: str = "1minute"):
    # ... existing ...
    # Fetch Nifty Data (Wrapper for Backward Compatibility)
    data = await upstox_service.get_instrument_history("NSE_INDEX|Nifty 50", interval)
    # Check for internal error reporting
    if isinstance(data, dict) and 'error' in data:
         return {"status": "error", "message": data['error'], "trace": data.get('trace')}
    if not data:
        return {"status": "error", "message": "Failed to fetch data (Unknown)"}
    return {"status": "success", "data": data}

@app.get("/market/history")
async def get_market_history(instrument_key: str, interval: str = "1minute"):
    """
    Fetch historical/intraday data for charting.
    """
    # Use get_intraday_candles for rich data (history + live + indicators)
    data = await upstox_service.get_intraday_candles(instrument_key, interval)
    if not data:
         return {"status": "error", "message": "No data found"}
    return {"status": "success", "data": data}



@app.get("/market/intraday")
async def get_intraday_data(instrument_key: str, interval: str = "1minute"):
    data = await upstox_service.get_intraday_candles(instrument_key, interval)
    if not data:
        return {"status": "error", "message": "Failed to fetch intraday data"}
    return {"status": "success", "data": data}


@app.get("/market/history")
async def get_market_history_generic(instrument_key: str, interval: str = "1minute"):
    # Generic Endpoint
    data = await upstox_service.get_instrument_history(instrument_key, interval)
    
    if isinstance(data, dict) and 'error' in data:
         return {"status": "error", "message": data['error'], "trace": data.get('trace')}
    if not data:
        return {"status": "error", "message": "Failed to fetch data"}
    return {"status": "success", "data": data}


@app.get("/aliceblue/option-chain")
async def get_alice_option_chain(index_name: str, expiry: str):
    """
    Fetch Option Chain from Alice Blue.
    """
    return alice_blue_service.get_option_chain(index_name, expiry)

@app.post("/aliceblue/place-order")
async def place_alice_order(transaction_type: str, instrument_token: str, quantity: int, price: float):
    """
    Place Order via Alice Blue (TradeMaster).
    """
    return alice_blue_service.place_order(transaction_type, instrument_token, quantity, price)

@app.get("/market/options/expiry")
async def get_expiry_dates(instrument_key: str):
    data = await upstox_service.get_expiry_dates(instrument_key)
    return {"data": data}

# --- Data Viewer Routes ---
@app.get("/data/upstox")
async def get_upstox_data(request: Request):
    # Fetch up to 100 recent documents from 'upstox_collection'
    # Note: Using request.app.mongodb to access the db instance from lifespan
    cursor = request.app.mongodb["upstox_collection"].find().sort("_id", -1).limit(100)
    documents = await cursor.to_list(length=100)
    
    # Convert ObjectId to string for JSON serialization
    for doc in documents:
        doc["_id"] = str(doc["_id"])
        
    return documents

# --- Mock Trading Routes ---

class MockOrderRequest(BaseModel):
    instrument_key: str
    quantity: int
    transaction_type: str
    trading_symbol: str | None = None

@app.post("/trade/mock/place")
async def place_mock_order(order: MockOrderRequest):
    return await mock_trade_service.place_order(order.dict())

@app.get("/trade/mock/positions")
async def get_mock_positions():
    return await mock_trade_service.get_positions()

@app.get("/trade/mock/exit/{trade_id}")
async def exit_mock_position(trade_id: str):
    return await mock_trade_service.exit_position(trade_id)

@app.get("/trade/mock/history")
async def get_mock_history():
    return await mock_trade_service.get_history()

# -------------------------------------------------------------------------
# ALICE BLUE ROUTES
# -------------------------------------------------------------------------

@app.get("/alice/auth/status")
async def get_alice_status():
    session = alice_blue_service.get_session()
    return {"status": "CONNECTED" if session else "DISCONNECTED"}

@app.get("/alice/market/options/chain")
async def get_alice_option_chain(instrument_key: str, expiry_date: str):
    return alice_blue_service.get_option_chain(instrument_key, expiry_date)
# --- Scanner Persistence Routes ---

class ScannerInstrumentItem(BaseModel):
    instrument_key: str
    name: str = ""
    exchange: str = ""
    segment: str = ""
    trading_symbol: str = ""
    mtf_enabled: bool = False

@app.get("/scanner/instruments")
async def get_scanner_instruments():
    # Switch to 'scanner_instruments_main'
    cursor = app.mongodb["scanner_instruments_main"].find().sort("name", 1)
    docs = await cursor.to_list(length=10000) 
    return {"data": [serialize_doc(doc) for doc in docs]}

@app.post("/scanner/instruments")
async def add_scanner_instrument(items: List[ScannerInstrumentItem]):
    import datetime
    operations = []
    from pymongo import UpdateOne
    
    for item in items:
        doc = item.dict()
        doc["added_at"] = datetime.datetime.now().isoformat()
        # Upsert based on instrument_key to avoid duplicates
        operations.append(
            UpdateOne(
                {"instrument_key": item.instrument_key},
                {"$set": doc},
                upsert=True
            )
        )
    
    if operations:
        await app.mongodb["scanner_instruments"].bulk_write(operations)
    
    return {"status": "success", "message": f"Processed {len(items)} instruments"}

@app.delete("/scanner/instruments/{instrument_key}")
async def remove_scanner_instrument(instrument_key: str):
    res = await app.mongodb["scanner_instruments"].delete_one({"instrument_key": instrument_key})
    return {"status": "success", "deleted_count": res.deleted_count}

    await app.mongodb["scanner_instruments"].delete_many({})
    return {"status": "success"}

def sanitize_nan(obj):
    if isinstance(obj, float):
        import math
        if math.isnan(obj) or math.isinf(obj):
            return 0  # Or None, but frontend handles 0 better for numbers
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    return obj

@app.get("/scanner/results")
async def get_scanner_results():
    try:
        cursor = app.mongodb["scanner_latest_results"].find({})
        results = await cursor.to_list(length=5000)
        for r in results:
            if "_id" in r: del r["_id"]
        
        clean_results = sanitize_nan(results)
        return {"data": clean_results}
    except Exception as e:
        print(f"Error fetching results: {e}")
        return {"data": []}

@app.post("/scanner/populate")
async def populate_scanner(index: str = "NIFTY 50"):
    from app.services.scanner_populate import ScannerPopulateService
    service = ScannerPopulateService(app.mongodb, upstox_service)
    return await service.populate_index(index)

@app.post("/scanner/populate_fno")
async def populate_scanner_fno():
    from app.services.scanner_populate import ScannerPopulateService
    service = ScannerPopulateService(app.mongodb, upstox_service)
    return await service.populate_from_fno()

class ScannerProcessRequest(BaseModel):
    interval: str = "1minute"
    instrument_keys: Optional[List[str]] = None # Optional: if empty, process all in DB
    mode: str = "combined" # 'combined', 'history', 'intraday'
    force_refresh: bool = False

@app.post("/scanner/process")
async def process_scanner_data(req: ScannerProcessRequest):
    # 1. Get Instruments
    if req.instrument_keys:
        keys = req.instrument_keys
    else:
        # Fetch from DB
        cursor = app.mongodb["scanner_instruments"].find({}, {"instrument_key": 1})
        docs = await cursor.to_list(length=10000)
        keys = [d["instrument_key"] for d in docs]

    if not keys:
        return {"data": []}

    # 2. Force Refresh: Delete existing results if requested
    if req.force_refresh:
        # print(f"Force Refresh: Deleting results for {len(keys)} instruments.")
        try:
            await app.mongodb["scanner_latest_results"].delete_many({"instrument_key": {"$in": keys}})
        except Exception as e:
            print(f"Error during force refresh delete: {e}")

    # 3. Bulk Fetch with Concurrency Control
    # Upstox Rate Limit is usually ~10 requests/sec or so for history? 
    # Actually V2 API history is generous but we should be careful.
    # Let's use a Semaphore.
    
    # import asyncio
    # limit = asyncio.Semaphore(1) # Removed for pure loop
    
    async def fetch_safe(key):
        # async with limit: # No semaphore needed in sequential loop
        try:
            # Use cached pivot data if we were doing a super optimized version, 
            # but get_intraday_candles handles it per call.
            # Optimization: We could pre-fetch daily candles for all keys in one go 
            # if there was a bulk API, but there isn't.
            import time
            t0 = time.time()
            data = await upstox_service.get_intraday_candles(key, req.interval)
            t1 = time.time()
            if (t1 - t0) > 2.0:
                print(f"SLOW FETCH: {key} took {t1 - t0:.2f}s")
            
            if data and "ltp" in data:
                data["instrument_key"] = key
                return data
            return None
        except Exception as e:
            print(f"Error processing {key}: {e}")
            return None

    # Sequential Loop
    results = []
    import time
    start_time = time.time()
    
    for k in keys:
        try:
             # Direct call, no semaphore needed since it's sequential
             # Use new full orchestration
             res = await upstox_service.process_instrument_full(k, req.interval, app.mongodb, req.mode)
             results.append(res)
        except Exception as e:
             print(f"Loop Error {k}: {e}")
             results.append(None)

    end_time = time.time()
    end_time = time.time()
    
    # Filter valid results
    valid_data = [r for r in results if r is not None]

    # Log missing/failed instruments
    successful_keys = {r.get("instrument_key") for r in valid_data if r}
    failed_keys = [k for k in keys if k not in successful_keys]
    if failed_keys:
        print(f"=== SCANNER: FAILED INSTRUMENTS ({len(failed_keys)}/{len(keys)}) ===")
        for fk in failed_keys:
            print(f"  FAILED: {fk}")
        print(f"=== END FAILED INSTRUMENTS ===")

    # 3. Save to Database (Persist Results)
    if valid_data:
        from pymongo import UpdateOne
        import datetime
        
        operations = []
        timestamp = datetime.datetime.now().isoformat()
        
        for item in valid_data:
            key = item.get("instrument_key")
            if key:
                # Add timestamp
                item["updated_at"] = timestamp
                operations.append(
                    UpdateOne(
                        {"instrument_key": key},
                        {"$set": item},
                        upsert=True
                    )
                )
        
        if operations:
            try:
                await app.mongodb["scanner_results"].bulk_write(operations)
                print(f"Saved {len(operations)} records to DB")
            except Exception as e:
                print(f"DB Save Error: {e}")
    
    print(f"Scanner Batch ({len(keys)} items, {len(valid_data)} success, {len(failed_keys)} failed) processed in {end_time - start_time:.2f} seconds.")
    
    return {"data": valid_data}
@app.post("/scanner/populate_fno")
async def populate_scanner_fno():
    """
    Populates scanner_instruments from the local fno_stocks collection
    """
    res = await scanner_populate.populate_from_fno()
    return res

@app.post("/scanner/fetch-fno")
async def fetch_fno_from_nse():
    """
    Downloads FNO list from NSE website and updates fno_stocks collection
    """
    res = await scanner_populate.fetch_fno_list()
    return res

@app.post("/scanner/fetch-master")
async def fetch_master_instruments():
    """
    Downloads Upstox Master Instrument List (NSE) and populates upstox_collection
    """
    res = await scanner_populate.fetch_master_instruments()
    return res
