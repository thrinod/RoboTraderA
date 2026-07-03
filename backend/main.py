import numpy as np
from datetime import datetime, timedelta
# Monkey-patch NumPy 2.0 compatibility for pandas-ta
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "float_"):
    np.float_ = float
if not hasattr(np, "int_"):
    np.int_ = int

from fastapi import FastAPI, Request, Header, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

# Load env vars first!
load_dotenv()

# Import services
from app.services.upstox_service import upstox_service
from app.services.trading_engine import trading_engine
from app.services.mock_trade_service import mock_trade_service
from app.services.alice_blue_service import alice_blue_service
from app.services.alice_blue_service import alice_blue_service
from app.services.scanner_populate import ScannerPopulateService
from app.services.charges_service import charges_service
from app.services.backtest_service import BacktestService
from app.services.telegram_service import telegram_service
from app.services.mirae_service import mirae_service

# Global Service Instances
scanner_populate = None
backtest_service = BacktestService(upstox_service)

import mock_npci_payment


MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "robotrader")

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
        telegram_service.set_db(app.mongodb)
        await telegram_service.load_config()
        
        # Initialize Scanner Populate Service
        global scanner_populate
        scanner_populate = ScannerPopulateService(app.mongodb, upstox_service)
        
        # Create Indexes for high performance (App Start "Cache" optimization)
        print("Ensuring Database Indexes for Scanner...")
        await app.mongodb["scanner_instruments_main"].create_index("instrument_key", unique=True)
        await app.mongodb["scanner_instruments_main"].create_index("name")
        await app.mongodb["scanner_instruments_fno"].create_index("instrument_key", unique=True)
        await app.mongodb["scanner_instruments_fno"].create_index("name")
        await app.mongodb["scanner_instruments_all"].create_index("instrument_key", unique=True)
        await app.mongodb["scanner_instruments_all"].create_index("name")
        await app.mongodb["upstox_collection"].create_index("trading_symbol")
        await app.mongodb["upstox_collection"].create_index("instrument_key")
        await app.mongodb["scanner_instruments"].create_index("SYMBOL")
        
        # Deduplicate scanner_results before creating unique index
        try:
            pipeline = [
                {"$group": {"_id": "$instrument_key", "count": {"$sum": 1}, "ids": {"$push": "$_id"}}},
                {"$match": {"count": {"$gt": 1}}}
            ]
            duplicates = await app.mongodb["scanner_results"].aggregate(pipeline).to_list(length=1000)
            if duplicates:
                print(f"Found {len(duplicates)} duplicate instrument keys in scanner_results. Cleaning up...")
                for doc in duplicates:
                    # Keep the first one, delete the rest
                    ids_to_delete = doc["ids"][1:]
                    await app.mongodb["scanner_results"].delete_many({"_id": {"$in": ids_to_delete}})
            
            # Drop existing non-unique index to avoid conflicts
            try:
                await app.mongodb["scanner_results"].drop_index("instrument_key_1")
            except Exception:
                pass
            
            await app.mongodb["scanner_results"].create_index("instrument_key", unique=True)
        except Exception as e:
            print(f"Warning: Could not create unique index on scanner_results: {e}")
            # Fallback to non-unique index if unique build still fails
            try:
                await app.mongodb["scanner_results"].create_index("instrument_key")
            except Exception:
                pass
            
        print("Indexes Ready.")
        
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
        
        # Auto-start Trading Engine if there are active deployments
        try:
            active_deployments = await app.mongodb["strategy_deployments"].count_documents({"status": "ACTIVE"})
            if active_deployments > 0:
                print(f"Lifespan: Found {active_deployments} active deployments. Starting Engine...")
                await trading_engine.start_trading()
            else:
                print("Lifespan: No active deployments found on startup.")
        except Exception as e:
            print(f"Lifespan: Failed to auto-start trading engine: {e}")

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

from starlette.requests import HTTPConnection

async def verify_app_token(conn: HTTPConnection, x_app_token: str = Header(None)):
    if conn.url.path.startswith("/ws/"):
        return
    if conn.url.path == "/process-payment" or conn.url.path.endswith("/process-payment"):
        return
        
    expected_token = os.getenv("APP_PASSWORD", "admin")
    if expected_token and x_app_token != expected_token:
        # For websockets, we could raise WebSocketException, but we already returned above
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid application token"
        )

app = FastAPI(lifespan=lifespan, dependencies=[Depends(verify_app_token)])
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://robo-webapp.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mock_npci_payment.router)

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
async def auth_callback(request: Request, code: str):
    token = await upstox_service.exchange_code_for_token(code)
    await upstox_service.save_token(request.app.mongodb, token)
    return {"message": "Authenticated successfully", "access_token": token}

# --- Database Viewer Routes ---

def serialize_doc(doc):
    """Convert ObjectId and datetime to string for JSON serialization"""
    if not doc:
        return doc
    for k, v in doc.items():
        if k == "_id":
            doc[k] = str(v)
        elif isinstance(v, datetime):
            doc[k] = v.isoformat()
        elif isinstance(v, list):
            doc[k] = [serialize_doc(i) if isinstance(i, dict) else i for i in v]
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
    
    import re
    safe_q = re.escape(q)
    regex = {"$regex": safe_q, "$options": "i"}
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
        query_filters.append({"mtf_enabled": True})
    
    query = {"$and": query_filters} if query_filters else {}
    
    # Prioritize NSE_EQ by sorting or by doing two fetches
    # For simplicity, we'll fetch NSE_EQ first, then the rest if limit not reached
    cursor = app.mongodb["upstox_collection"].find(query).sort([("exchange", -1), ("name", 1)]).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    # Map 'NSE' to 'NSE_EQ' if needed for display consistency
    serialized = []
    for doc in docs:
        d = serialize_doc(doc)
        serialized.append(d)
        
    return {"data": serialized}

@app.get("/market/instruments/types")
async def get_instrument_types():
    """Returns unique instrument types from upstox_collection"""
    types = await app.mongodb["upstox_collection"].distinct("instrument_type")
    # Filter out empty or None
    valid_types = [t for t in types if t]
    return {"data": sorted(valid_types)}

class TokenRequest(BaseModel):
    token: str
    algo_name: Optional[str] = None

class OrderItem(BaseModel):
    instrument_key: str
    quantity: int
    transaction_type: str
    order_type: str = "MARKET"
    price: float = 0.0
    trading_symbol: Optional[str] = None

class BulkOrderRequest(BaseModel):
    orders: List[OrderItem]

@app.post("/auth/token")
async def set_token(request: Request, token_req: TokenRequest):
    is_valid, error_msg = upstox_service.verify_token(token_req.token)
    if is_valid:
        await upstox_service.save_token(request.app.mongodb, token_req.token, token_req.algo_name)
        return {"message": "Token verified and connected", "status": "success", "upstox_status": "Connected"}
    else:
        return {"message": f"Invalid Token: {error_msg}", "status": "error", "upstox_status": "Disconnected"}

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
    async def process_order(order):
        return await upstox_service.place_order(
            instrument_key=order.instrument_key,
            quantity=order.quantity,
            transaction_type=order.transaction_type,
            order_type=order.order_type,
            price=order.price
        )
    
    import asyncio
    res_list = await asyncio.gather(*(process_order(o) for o in req.orders))
    
    results = [{"key": o.instrument_key, "result": r} for o, r in zip(req.orders, res_list)]
    
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

@app.get("/user/charges")
def get_user_charges():
    """Calculate actual F&O charges for today's completed trades."""
    try:
        trades = upstox_service.get_trade_book()
        result = charges_service.calculate_charges(trades)
        return {"status": "success", "data": result}
    except Exception as e:
        print(f"Error calculating charges: {e}")
        return {"status": "error", "message": str(e), "data": {"total": {"grand_total": 0, "trade_count": 0, "order_count": 0}, "trades": []}}

@app.get("/market/options/chain")
async def get_option_chain(instrument_key: str, expiry_date: str):
    # instrument_key example: NSE_INDEX|Nifty 50
    # expiry_date example: 2025-12-28
    spot_quotes = upstox_service.get_market_quotes([instrument_key])
    spot_price = 0.0
    spot_change = 0.0
    spot_pchange = 0.0
    
    norm_key = instrument_key.replace(':', '|')
    spot_data = spot_quotes.get(norm_key) or spot_quotes.get(instrument_key)
    if spot_data:
        spot_price = spot_data.get('ltp', 0.0)
        spot_change = spot_data.get('change', 0.0)
        spot_pchange = spot_data.get('change_percent', 0.0)
    else:
        spot_price = await upstox_service.get_spot_price(instrument_key) or 0.0
        
    result = await upstox_service.get_option_chain(instrument_key, expiry_date)
    if not isinstance(result, dict):
        result = {}
    return {
        "data": result.get('chain', []),
        "spot_price": spot_price,
        "spot_change": spot_change,
        "spot_pchange": spot_pchange,
        "totals": result.get('totals', {'ce': 0, 'pe': 0})
    }

import json
import asyncio

@app.websocket("/ws/option-chain")
async def ws_option_chain(websocket: WebSocket):
    await websocket.accept()
    
    task = None
    
    async def fetch_and_push(instrument_key: str, expiry_date: str, interval: float):
        while True:
            try:
                # Lazy load token if needed
                if not upstox_service.access_token:
                    await upstox_service.load_token(app.mongodb)
                
                spot_quotes = upstox_service.get_market_quotes([instrument_key])
                spot_price = 0.0
                spot_change = 0.0
                spot_pchange = 0.0
                norm_key = instrument_key.replace(':', '|')
                spot_data = spot_quotes.get(norm_key) or spot_quotes.get(instrument_key)
                if spot_data:
                    spot_price = spot_data.get('ltp', 0.0)
                    spot_change = spot_data.get('change', 0.0)
                    spot_pchange = spot_data.get('change_percent', 0.0)
                else:
                    spot_price = await upstox_service.get_spot_price(instrument_key) or 0.0

                result = await upstox_service.get_option_chain(instrument_key, expiry_date)
                if not isinstance(result, dict):
                    result = {}
                
                payload = {
                    "status": "success",
                    "data": {
                        "chain": result.get('chain', []),
                        "spot_price": spot_price or 0,
                        "spot_change": spot_change,
                        "spot_pchange": spot_pchange,
                        "totals": result.get('totals', {'ce': 0, 'pe': 0})
                    }
                }
                await websocket.send_json(payload)
            except Exception as e:
                print(f"Error in ws_option_chain fetch_and_push loop: {e}")
                try:
                    await websocket.send_json({"status": "error", "message": str(e)})
                except:
                    pass
            await asyncio.sleep(interval)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "subscribe":
                    instrument_key = msg.get("instrument_key")
                    expiry_date = msg.get("expiry_date")
                    interval = float(msg.get("interval", 2.0))
                    
                    if not instrument_key or not expiry_date:
                        await websocket.send_json({"status": "error", "message": "Missing instrument_key or expiry_date"})
                        continue
                    
                    if task:
                        task.cancel()
                    
                    task = asyncio.create_task(fetch_and_push(instrument_key, expiry_date, interval))
                    print(f"Option Chain WS Subscribed to {instrument_key} | Expiry: {expiry_date}")
            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "message": "Invalid JSON"})
            except Exception as e:
                await websocket.send_json({"status": "error", "message": str(e)})
    except WebSocketDisconnect:
        print("Option Chain WS Client Disconnected")
    finally:
        if task:
            task.cancel()

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

# --- Checklist Routes ---
class ChecklistRuleItem(BaseModel):
    id: str
    title: str
    desc: str = ""
    weight: int
    checked: bool = False

class ChecklistRulesSaveRequest(BaseModel):
    rules: List[ChecklistRuleItem]

class ChecklistThresholdRequest(BaseModel):
    threshold: int

class JournalRuleAudit(BaseModel):
    title: str
    weight: int

class JournalEntryModel(BaseModel):
    id: str
    timestamp: str
    symbol: str
    name: str
    action: str
    qty: int
    price: float
    orderType: str
    product: str
    mode: str
    score: int
    threshold: int
    passedRules: List[JournalRuleAudit]
    failedRules: List[JournalRuleAudit]
    status: str
    errorMsg: Optional[str] = None

@app.get("/checklist/rules")
async def get_checklist_rules():
    doc = await app.mongodb["checklist_settings"].find_one({"key": "rules"})
    if doc:
        return {"rules": doc.get("rules", [])}
    return {"rules": []}

@app.post("/checklist/rules")
async def save_checklist_rules(req: ChecklistRulesSaveRequest):
    await app.mongodb["checklist_settings"].update_one(
        {"key": "rules"},
        {"$set": {"rules": [r.dict() for r in req.rules]}},
        upsert=True
    )
    return {"status": "success"}

@app.get("/checklist/threshold")
async def get_checklist_threshold():
    doc = await app.mongodb["checklist_settings"].find_one({"key": "threshold"})
    if doc:
        return {"threshold": doc.get("threshold", 70)}
    return {"threshold": 70}

@app.post("/checklist/threshold")
async def save_checklist_threshold(req: ChecklistThresholdRequest):
    await app.mongodb["checklist_settings"].update_one(
        {"key": "threshold"},
        {"$set": {"threshold": req.threshold}},
        upsert=True
    )
    return {"status": "success"}

@app.get("/checklist/journal")
async def get_checklist_journal():
    cursor = app.mongodb["checklist_journal"].find({}).sort("_id", -1).limit(100)
    docs = await cursor.to_list(length=100)
    return {"data": [serialize_doc(doc) for doc in docs]}

@app.post("/checklist/journal")
async def add_journal_entry(entry: JournalEntryModel):
    doc = entry.dict()
    await app.mongodb["checklist_journal"].insert_one(doc)
    return {"status": "success"}

@app.delete("/checklist/journal")
async def clear_checklist_journal():
    await app.mongodb["checklist_journal"].delete_many({})
    return {"status": "success"}

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

# --- Mirae F&O Endpoints ---

@app.get("/mirae/options/expiry")
async def get_mirae_expiry_dates(instrument_key: str):
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if config and config.get("enabled"):
        mirae_access_token = config.get("mirae_access_token")
        api_key = config.get("mirae_api_key")
        success, msg = mirae_service.initialize(mirae_access_token, api_key)
        if success:
            expiries = mirae_service.get_expiry_dates(instrument_key)
            if expiries:
                return {"data": expiries}
                
    # Ultimate fallback: Dummy expiries
    import datetime
    dates = []
    today = datetime.date.today()
    for i in range(1, 35):
        d = today + datetime.timedelta(days=i)
        if d.weekday() == 3: # Thursday
            dates.append(d.strftime("%Y-%m-%d"))
    return {"data": dates[:5]}


@app.get("/mirae/options/chain")
async def get_mirae_option_chain(instrument_key: str, expiry_date: str):
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if config and config.get("enabled"):
        mirae_access_token = config.get("mirae_access_token")
        api_key = config.get("mirae_api_key")
        success, msg = mirae_service.initialize(mirae_access_token, api_key)
        if success:
            chain_data = mirae_service.get_option_chain(instrument_key, expiry_date)
            if chain_data and chain_data.get("chain"):
                return {
                    "data": chain_data.get("chain", []),
                    "spot_price": chain_data.get("spot_price", 0.0),
                    "spot_change": chain_data.get("spot_change", 0.0),
                    "spot_pchange": chain_data.get("spot_pchange", 0.0),
                    "totals": chain_data.get("totals", {"ce": 0, "pe": 0})
                }


    # Ultimate fallback: Generate mock option chain
    spot_price = 24000.0 if "nifty 50" in instrument_key.lower() or "nifty" in instrument_key.lower() else (52000.0 if "bank" in instrument_key.lower() else 20000.0)
    
    mock_chain = []
    base_strike = int((spot_price // 50) * 50) if "nifty 50" in instrument_key.lower() or "nifty" in instrument_key.lower() else int((spot_price // 100) * 100)
    step = 50 if "nifty 50" in instrument_key.lower() or "nifty" in instrument_key.lower() else 100
    
    for i in range(-15, 16):
        strike = base_strike + (i * step)
        c_dist = (spot_price - strike) / step
        c_price = max(2.0, 100.0 + c_dist * 40.0)
        c_oi = max(1000, int(100000 - abs(c_dist) * 8000))
        
        p_dist = (strike - spot_price) / step
        p_price = max(2.0, 100.0 + p_dist * 40.0)
        p_oi = max(1000, int(100000 - abs(p_dist) * 8000))
        
        # Generate some mock price changes
        c_close = round(c_price * 0.95, 2)
        c_change = round(c_price - c_close, 2)
        c_pchange = round((c_change / c_close) * 100, 2) if c_close > 0 else 0.0

        mock_chain.append({
            'strike_price': float(strike),
            'instrument_type': 'CE',
            'instrument_key': f"MOCK:NIFTY:{strike}:CE",
            'last_price': round(c_price, 2),
            'open_interest': c_oi,
            'volume': 5000,
            'price_change': c_change,
            'pchange': c_pchange,
            'bid_price': round(c_price - 0.1, 2),
            'ask_price': round(c_price + 0.1, 2),
            'low_price': round(c_price * 0.9, 2),
            'iv': 15.0,
            'delta': 0.5,
            'oi_value': c_oi * c_price,
            'lot_size': 50
        })
        
        p_close = round(p_price * 1.05, 2)
        p_change = round(p_price - p_close, 2)
        p_pchange = round((p_change / p_close) * 100, 2) if p_close > 0 else 0.0

        mock_chain.append({
            'strike_price': float(strike),
            'instrument_type': 'PE',
            'instrument_key': f"MOCK:NIFTY:{strike}:PE",
            'last_price': round(p_price, 2),
            'open_interest': p_oi,
            'volume': 5000,
            'price_change': p_change,
            'pchange': p_pchange,
            'bid_price': round(p_price - 0.1, 2),
            'ask_price': round(p_price + 0.1, 2),
            'low_price': round(p_price * 0.9, 2),
            'iv': 15.0,
            'delta': -0.5,
            'oi_value': p_oi * p_price,
            'lot_size': 50
        })
        
    return {
        "data": mock_chain,
        "spot_price": spot_price,
        "spot_change": 120.50,
        "spot_pchange": 0.50,
        "totals": {"ce": 1500000, "pe": 1400000}
    }



@app.post("/mirae/trade/place_orders")
async def place_mirae_orders(req: BulkOrderRequest):
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if not config or not config.get("enabled"):
        return {"status": "error", "message": "Mirae Asset integration is disabled"}
        
    mirae_access_token = config.get("mirae_access_token")
    api_key = config.get("mirae_api_key")
    
    success, msg = mirae_service.initialize(mirae_access_token, api_key)
    if not success:
        return {"status": "error", "message": f"Login failed: {msg}"}
        
    results = []
    for order in req.orders:
        # Prefer instrument_key when it has an "EXCHANGE:SYMBOL" prefix (set by frontend adapter)
        # Fall back to trading_symbol for backwards compatibility
        ik = order.instrument_key or ""
        ts = order.trading_symbol or ""
        raw_symbol = ik if ":" in ik else (ts if ts else ik)
        print(f"[DEBUG place_mirae_orders] instrument_key={ik}, trading_symbol={ts}, raw_symbol={raw_symbol}")

        # Detect exchange from prefix (e.g. "BSE:SENSEX-02Jul2026-77400-CE" or "NFO:NIFTY...")
        if ":" in raw_symbol:
            exchange_prefix, symbol = raw_symbol.split(":", 1)
        else:
            exchange_prefix = ""
            symbol = raw_symbol

        # Use the exchange prefix directly as the source exchange for dynamic resolution
        # EXCHANGE_MAP in mirae_service handles: NFO->NFO, BFO->BFO, NSE->NFO, BSE->BFO, etc.
        src_exchange = exchange_prefix if exchange_prefix else "NFO"
        order_symbol, order_exchange = mirae_service._translate_position_symbol(symbol, src_exchange)

        print(f"[DEBUG place_mirae_orders] symbol_sent={order_symbol}, exchange={order_exchange}")

        # mStock valid order types: MARKET, LIMIT, SL, SL-M  (same for all exchanges, no 'MKT' alias)
        raw_type = (order.order_type or "MARKET").upper()
        if raw_type in ("MKT", "MKT_ORDER"):
            o_type = "MARKET"
        elif raw_type in ("LMT",):
            o_type = "LIMIT"
        else:
            o_type = raw_type  # MARKET, LIMIT, SL, SL-M passed through as-is

        # Product: use what is provided or default to NRML (standard for F&O)
        product = getattr(order, "product", None) or "NRML"

        res = mirae_service.place_order(
            variety="regular",
            tradingsymbol=order_symbol,
            exchange=order_exchange,
            transaction_type=order.transaction_type,
            order_type=o_type,
            quantity=order.quantity,
            product=product,
            validity="DAY",
            price=order.price or 0.0,
            trigger_price=0.0
        )
        print(f"[DEBUG place_mirae_orders] mconnect_response={res}")
        results.append({"key": order.instrument_key, "result": res})
        
    return {"status": "completed", "results": results}


@app.post("/mirae/trade/cancel_all")
async def cancel_mirae_orders():
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if not config or not config.get("enabled"):
        return {"status": "error", "message": "Mirae Asset integration is disabled"}
        
    mirae_access_token = config.get("mirae_access_token")
    api_key = config.get("mirae_api_key")
    
    success, msg = mirae_service.initialize(mirae_access_token, api_key)
    if not success:
        return {"status": "error", "message": f"Login failed: {msg}"}
        
    res_success, res_data = mirae_service.cancel_all_orders()
    if not res_success:
        return {"status": "error", "message": f"Cancel all failed: {res_data}"}
    return {"status": "success", "data": res_data}


@app.post("/mirae/trade/square_off")
async def square_off_mirae_all():
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if not config or not config.get("enabled"):
        return {"status": "error", "message": "Mirae Asset integration is disabled"}
        
    mirae_access_token = config.get("mirae_access_token")
    api_key = config.get("mirae_api_key")
    
    success, msg = mirae_service.initialize(mirae_access_token, api_key)
    if not success:
        return {"status": "error", "message": f"Login failed: {msg}"}
        
    res = mirae_service.square_off_all_positions()
    return res


@app.post("/mirae/trade/exit")
async def exit_mirae_position(req: ExitRequest):
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if not config or not config.get("enabled"):
        return {"status": "error", "message": "Mirae Asset integration is disabled"}
        
    mirae_access_token = config.get("mirae_access_token")
    api_key = config.get("mirae_api_key")
    
    success, msg = mirae_service.initialize(mirae_access_token, api_key)
    if not success:
        return {"status": "error", "message": f"Login failed: {msg}"}
        
    p_success, p_data = mirae_service.get_net_position()
    if not p_success:
        return {"status": "error", "message": f"Failed to get positions: {p_data}"}

    # Extract positions from nested data.net structure
    positions_list = []
    if isinstance(p_data, list):
        positions_list = p_data
    elif isinstance(p_data, dict):
        data_val = p_data.get("data", p_data)
        if isinstance(data_val, dict):
            positions_list = data_val.get("net", []) or []
        elif isinstance(data_val, list):
            positions_list = data_val

    req_key = req.instrument_key
    # Strip exchange prefix for matching
    if ":" in req_key:
        req_key = req_key.split(":", 1)[1]
        
    for pos in positions_list:
        symbol = pos.get("tradingsymbol") or pos.get("tradingSymbol") or pos.get("symbol") or ""
        instrument_token = str(pos.get("instrument_token", ""))
        # Match by trading symbol, instrument token, or partial key
        if symbol == req_key or instrument_token == req_key or req_key in symbol:
            buy_qty = int(pos.get("buy_quantity", 0) or 0)
            sell_qty = int(pos.get("sell_quantity", 0) or 0)
            net_qty = buy_qty - sell_qty
            if net_qty != 0:
                prod = pos.get("product") or pos.get("productType") or "MIS"
                exchange = pos.get("exchange", "NFO")
                side = "SELL" if net_qty > 0 else "BUY"
                abs_qty = abs(net_qty)
                print(f"[DEBUG exit_mirae] Exiting: symbol={symbol}, exchange={exchange}, qty={abs_qty}, side={side}")
                res = mirae_service.square_off_position(symbol, abs_qty, side, prod, exchange)
                print(f"[DEBUG exit_mirae] Response: {res}")
                return {"status": "success", "result": res}
            else:
                return {"status": "error", "message": f"Position {symbol} already closed (net qty = 0)"}
                
    return {"status": "error", "message": f"No open position found for {req.instrument_key}"}

class MiraeQuoteRequest(BaseModel):
    instrument_keys: list[str]

@app.post("/mirae/market/quotes")
async def get_mirae_quotes(req: MiraeQuoteRequest):
    if not req.instrument_keys:
        return {"data": {}}
        
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if config and config.get("enabled"):
        mirae_access_token = config.get("mirae_access_token")
        api_key = config.get("mirae_api_key")
        success, msg = mirae_service.initialize(mirae_access_token, api_key)
        if success:
            mirae_symbols = []
            symbol_map = {}
            for key in req.instrument_keys:
                sym = "NSE:Nifty 50" if "nifty 50" in key.lower() or "nifty" in key.lower() else "NSE:Nifty Bank"
                if ":" in key:
                    sym = key
                mirae_symbols.append(sym)
                symbol_map[sym] = key
                
            try:
                # Cache previous close prices for requested symbols
                missing_symbols = [s for s in mirae_symbols if s not in mirae_service.prev_close_cache]
                if missing_symbols:
                    try:
                        ohlc_res = mirae_service.mconnect.get_ohlc(missing_symbols)
                        if ohlc_res.status_code == 200:
                            ohlc_data = ohlc_res.json().get("data", {})
                            if isinstance(ohlc_data, dict):
                                for sym_key, sym_data in ohlc_data.items():
                                    close_val = sym_data.get("close")
                                    if close_val is None:
                                        close_val = sym_data.get("ohlc", {}).get("close")
                                    if close_val is not None:
                                        mirae_service.prev_close_cache[sym_key] = float(close_val)
                    except Exception as e:
                        print(f"Error fetching quote OHLC close cache: {e}")

                res = mirae_service.mconnect.get_ltp(mirae_symbols)
                if res.status_code == 200:
                    data = res.json()
                    quotes_data = {}
                    res_data = data.get("data", {}) if isinstance(data, dict) else {}
                    if not res_data:
                        items = data if isinstance(data, list) else (data.get("data", []) or data.get("result", []) or [])
                        res_data = {item.get("symbol") or item.get("tradingSymbol") or item.get("tradingsymbol"): item for item in items if item}
                        
                    for sym, item in res_data.items():
                        req_key = symbol_map.get(sym) or sym
                        ltp = float(item.get("last_price") or item.get("ltp") or 0.0)
                        close = mirae_service.prev_close_cache.get(sym, float(item.get("close") or 0.0))
                        
                        price_change = round(ltp - close, 2) if close > 0 else 0.0
                        pchange = round((price_change / close) * 100, 2) if close > 0 else 0.0
                        
                        quotes_data[req_key] = {
                            "ltp": ltp,
                            "close": close,
                            "price_change": price_change,
                            "pchange": pchange
                        }
                    return {"data": quotes_data}
            except Exception as e:
                print(f"Error fetching Mirae quotes: {e}")

    mock_quotes = {}
    for key in req.instrument_keys:
        ltp = 24000.0 if "nifty 50" in key.lower() or "nifty" in key.lower() else 52000.0
        close = ltp * 0.99
        change = round(ltp - close, 2)
        pchange = round((change / close) * 100, 2)
        mock_quotes[key] = {
            "ltp": ltp, 
            "close": close,
            "price_change": change,
            "pchange": pchange
        }
    return {"data": mock_quotes}


@app.get("/mirae/user/charges")
async def get_mirae_user_charges():
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if not config or not config.get("enabled"):
        return {"status": "error", "message": "Mirae Asset integration is disabled"}
        
    mirae_access_token = config.get("mirae_access_token")
    api_key = config.get("mirae_api_key")
    
    success, msg = mirae_service.initialize(mirae_access_token, api_key)
    if not success:
        return {"status": "error", "message": f"Login failed: {msg}"}
        
    try:
        t_success, trades_data = mirae_service.get_trade_book()
        if not t_success:
            return {"status": "error", "message": f"Failed to get trade book: {trades_data}"}
            
        mapped_trades = []
        trades_list = []
        if isinstance(trades_data, list):
            trades_list = trades_data
        elif isinstance(trades_data, dict):
            trades_list = trades_data.get("trades", []) or trades_data.get("data", []) or []
            
        for t in trades_list:
            mapped_trades.append({
                "order_id": str(t.get("orderId") or t.get("ordId") or t.get("order_id") or ""),
                "trading_symbol": str(t.get("tradingSymbol") or t.get("tsym") or t.get("trading_symbol") or ""),
                "transaction_type": str(t.get("transactionType") or t.get("transType") or t.get("transaction_type") or "BUY").upper(),
                "quantity": int(t.get("quantity") or t.get("fillQty") or t.get("qty") or 0),
                "average_price": float(t.get("averagePrice") or t.get("fillPrc") or t.get("price") or t.get("average_price") or 0.0),
            })
            
        result = charges_service.calculate_charges(mapped_trades)
        return {"status": "success", "data": result}
    except Exception as e:
        print(f"Error calculating Mirae charges: {e}")
        return {"status": "error", "message": str(e), "data": {"total": {"grand_total": 0, "trade_count": 0, "order_count": 0}, "trades": []}}

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

@app.post("/trade/mock/place_orders")
async def place_mock_orders(req: BulkOrderRequest):
    import asyncio
    
    async def process_mock(o):
        return await mock_trade_service.place_order({
            "instrument_key": o.instrument_key,
            "quantity": o.quantity,
            "transaction_type": o.transaction_type,
            "order_type": o.order_type,
            "price": o.price
        })
        
    res_list = await asyncio.gather(*(process_mock(o) for o in req.orders))
    results = [{"key": o.instrument_key, "result": r} for o, r in zip(req.orders, res_list)]
    return {"status": "completed", "results": results}

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
async def get_scanner_instruments(type: str = "main"):
    collection_name = "scanner_instruments_main"
    if type == "fno":
        collection_name = "scanner_instruments_fno"
    elif type == "all":
        collection_name = "scanner_instruments_all"
    
    cursor = app.mongodb[collection_name].find().sort("name", 1)
    docs = await cursor.to_list(length=10000) 
    return {"data": [serialize_doc(doc) for doc in docs]}

@app.post("/scanner/instruments")
async def add_scanner_instrument(items: List[ScannerInstrumentItem], type: str = "main"):
    collection_name = "scanner_instruments_main"
    if type == "fno":
        collection_name = "scanner_instruments_fno"
    elif type == "all":
        collection_name = "scanner_instruments_all"
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
        await app.mongodb[collection_name].bulk_write(operations)
    
    return {"status": "success", "message": f"Processed {len(items)} instruments"}

@app.delete("/scanner/instruments/{instrument_key}")
async def remove_scanner_instrument(instrument_key: str, type: str = "main"):
    collection_name = "scanner_instruments_main"
    if type == "fno":
        collection_name = "scanner_instruments_fno"
    elif type == "all":
        collection_name = "scanner_instruments_all"
    res = await app.mongodb[collection_name].delete_one({"instrument_key": instrument_key})
    return {"status": "success", "deleted_count": res.deleted_count}

@app.delete("/scanner/instruments")
async def clear_scanner_instruments(type: str = "main"):
    collection_name = "scanner_instruments_main"
    if type == "fno":
        collection_name = "scanner_instruments_fno"
    elif type == "all":
        collection_name = "scanner_instruments_all"
    await app.mongodb[collection_name].delete_many({})
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
async def get_scanner_results(type: Optional[str] = None):
    try:
        query_filter = {}
        if type in ["main", "fno", "all"]:
            # Fetch valid instrument keys for the specified scanner page type
            collection_name = f"scanner_instruments_{type}"
            instruments_cursor = app.mongodb[collection_name].find({}, {"instrument_key": 1})
            valid_instruments = await instruments_cursor.to_list(length=10000)
            valid_keys = [doc["instrument_key"] for doc in valid_instruments if doc.get("instrument_key")]
            query_filter = {"instrument_key": {"$in": valid_keys}}
            
        # Project to exclude massive unused arrays (candles and indicators_series)
        # This reduces data loading and serialization time by 40x!
        cursor = app.mongodb["scanner_results"].find(
            query_filter, 
            {"candles": 0, "indicators_series": 0}
        )
        results = await cursor.to_list(length=10000)
        
        # Gather all unique instrument keys to lookup their name/trading_symbol/exchange
        keys = list(set([r["instrument_key"] for r in results if r.get("instrument_key")]))
        
        lookup = {}
        if keys:
            # Query directly from the master upstox_collection (super fast due to index)
            # Only project name, trading_symbol, exchange, instrument_key to save memory and CPU
            master_cursor = app.mongodb["upstox_collection"].find(
                {"instrument_key": {"$in": keys}},
                {"name": 1, "trading_symbol": 1, "exchange": 1, "instrument_key": 1}
            )
            master_list = await master_cursor.to_list(length=10000)
            lookup = {m["instrument_key"]: m for m in master_list}
        
        for r in results:
            if "_id" in r: del r["_id"]
            k = r.get("instrument_key")
            if k in lookup:
                r["name"] = lookup[k].get("name", "")
                r["trading_symbol"] = lookup[k].get("trading_symbol", "")
                r["exchange"] = lookup[k].get("exchange", "")
        
        clean_results = sanitize_nan(results)
        return {"data": clean_results}
    except Exception as e:
        print(f"Error fetching results: {e}")
        import traceback
        traceback.print_exc()
        return {"data": []}

@app.post("/scanner/populate")
async def populate_scanner(index: str = "NIFTY 50"):
    from app.services.scanner_populate import ScannerPopulateService
    service = ScannerPopulateService(app.mongodb, upstox_service)
    return await service.populate_index(index)

@app.post("/scanner/populate_fno")
async def populate_scanner_fno_route():
    service = ScannerPopulateService(app.mongodb, upstox_service)
    return await service.populate_from_fno()

@app.post("/scanner/populate_all")
async def populate_scanner_all_route():
    service = ScannerPopulateService(app.mongodb, upstox_service)
    return await service.populate_all_stocks()

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
        # Fetch from DB (Always use scanner_instruments_main)
        cursor = app.mongodb["scanner_instruments_main"].find({}, {"instrument_key": 1})
        docs = await cursor.to_list(length=10000)
        keys = [d["instrument_key"] for d in docs]

    if not keys:
        return {"data": []}

    # 2. Force Refresh: Delete existing results if requested
    if req.force_refresh:
        try:
            # Delete from the unified results collection
            await app.mongodb["scanner_results"].delete_many({"instrument_key": {"$in": keys}})
            # Also clear candles cache to ensure fresh data fetch
            await app.mongodb["instrument_candles"].delete_many({"instrument_key": {"$in": keys}})
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

    # Concurrent Processing with asyncio.gather
    import asyncio
    import time
    start_time = time.time()
    
    tasks = [
        upstox_service.process_instrument_full(
            k, req.interval, app.mongodb, req.mode, force=req.force_refresh
        )
        for k in keys
    ]
    
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = []
    for i, res in enumerate(raw_results):
        if isinstance(res, Exception):
            print(f"Concurrency Loop Error for {keys[i]}: {res}")
            results.append(None)
        else:
            results.append(res)
            
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
    
    print(f"Scanner Batch ({len(keys)} items, {len(valid_data)} success, {len(failed_keys)} failed) processed in {time.time() - start_time:.2f} seconds.")
    return {"data": valid_data}

class BacktestRequest(BaseModel):
    instrument_key: Optional[str] = None
    interval: Optional[str] = "15minute"
    days_back: int = 30
    stop_loss: float = 1.0
    take_profit: float = 2.0
    is_advanced: bool = False
    execution_plan: Optional[list] = None
    saved_strategies: Optional[list] = None
    trade_type: str = "LONG"
    trade_instrument_key: Optional[str] = None
    use_intraday: bool = False
    trailing_sl: bool = False
    trailing_sl_trigger_pct: float = 40.0 # Percentage profit to trigger move to BE

class DeployStrategyRequest(BacktestRequest):
    deployment_mode: str = "MOCK" # "MOCK" or "LIVE"
    quantity_type: str = "MANUAL" # "MANUAL" or "CAPITAL" or "PERCENTAGE"
    quantity: int = 1
    capital_to_use: float = 0.0
    capital_percentage: float = 0.0
    lot_size: int = 1

@app.post("/backtest/run")
async def run_backtest(req: BacktestRequest):
    # Determine which instrument to use for the single-leg logic bridge
    primary_instrument = req.instrument_key
    primary_interval = req.interval
    
    if req.is_advanced and req.execution_plan and len(req.execution_plan) > 0:
        primary_instrument = req.execution_plan[0].get("leg")
        primary_interval = req.execution_plan[0].get("timeframe")
        
    if not primary_instrument:
        raise HTTPException(status_code=400, detail="No instrument key provided.")
        
    print(f"Running backtest for {primary_instrument} | Interval: {primary_interval} | Days: {req.days_back} | Intraday: {req.use_intraday}")
    
    # We pass the primary instrument to our existing service until phase 3
    # Phase 3 will involve passing the full execution_plan and saved_strategies to the service
    result = await backtest_service.run_strategy(
        primary_instrument, 
        primary_interval, 
        req.days_back,
        req.stop_loss,
        req.take_profit,
        req.is_advanced,
        req.execution_plan,
        req.saved_strategies,
        req.trade_type,
        req.trade_instrument_key,
        use_intraday=req.use_intraday
    )
    return result

@app.post("/deploy/strategy")
async def deploy_strategy(req: DeployStrategyRequest):
    import datetime
    
    primary_instrument = req.instrument_key
    if req.is_advanced and req.execution_plan and len(req.execution_plan) > 0:
        primary_instrument = req.execution_plan[0].get("leg")
        
    doc = req.dict()
    doc["status"] = "ACTIVE"
    doc["primary_instrument"] = primary_instrument
    doc["deployed_at"] = datetime.datetime.now()
    
    res = await app.mongodb["strategy_deployments"].insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    
    # Ensure the trading engine is started when a strategy is deployed
    await trading_engine.start_trading()
    
    # Telegram Notification for Deployment
    try:
        # Get live resolved details
        primary_details = await get_live_instrument_details(primary_instrument)
        resolved_symbol = primary_details.get("trading_symbol") or primary_details.get("display_name") or primary_instrument
        if primary_details.get("name") and primary_details.get("name") != resolved_symbol:
            resolved_symbol = f"{resolved_symbol} ({primary_details.get('name')})"
            
        primary_option_info = format_live_option_info(primary_details)
        primary_option_block = f"{primary_option_info}" if primary_option_info else ""
        
        trade_option_block = ""
        trade_instrument = req.trade_instrument_key
        if trade_instrument and trade_instrument != primary_instrument:
            trade_details = await get_live_instrument_details(trade_instrument)
            trade_resolved = trade_details.get("trading_symbol") or trade_details.get("display_name") or trade_instrument
            if trade_details.get("name") and trade_details.get("name") != trade_resolved:
                trade_resolved = f"{trade_resolved} ({trade_details.get('name')})"
                
            trade_option_info = format_live_option_info(trade_details)
            trade_option_block = (
                f"\n<b>Trade Instrument:</b> {trade_resolved}\n"
                f"<b>Trade Key:</b> {trade_instrument}\n"
                f"{trade_option_info}"
            )
            
        legs_block = ""
        if req.is_advanced and req.execution_plan:
            legs_block = "\n<b>Execution Plan Legs:</b>\n"
            for i, leg in enumerate(req.execution_plan):
                leg_key = leg.get("leg")
                leg_timeframe = leg.get("timeframe")
                if leg_key:
                    leg_details = await get_live_instrument_details(leg_key)
                    leg_resolved = leg_details.get("trading_symbol") or leg_details.get("display_name") or leg_key
                    if leg_details.get("name") and leg_details.get("name") != leg_resolved:
                        leg_resolved = f"{leg_resolved} ({leg_details.get('name')})"
                        
                    leg_opt_info = format_live_option_info(leg_details)
                    leg_opt_indented = "\n".join([f"  {line}" for line in leg_opt_info.strip().split("\n") if line])
                    leg_opt_formatted = f"\n{leg_opt_indented}" if leg_opt_indented else ""
                    legs_block += f"• <b>Leg {i+1}:</b> {leg_resolved} ({leg_timeframe}){leg_opt_formatted}\n"

        msg = (
            f"🤖 <b>Strategy Deployed!</b>\n\n"
            f"<b>Instrument:</b> {resolved_symbol}\n"
            f"<b>Key:</b> {primary_instrument}\n"
            f"{primary_option_block}"
            f"{trade_option_block}"
            f"{legs_block}"
            f"\n<b>Mode:</b> {req.deployment_mode}\n"
            f"<b>Quantity:</b> {req.quantity}\n"
            f"<b>Interval:</b> {req.interval}\n"
            f"<b>Type:</b> {'Advanced' if req.is_advanced else 'Simple'}\n"
            f"<b>Status:</b> ACTIVE (Monitoring...)"
        )
        await telegram_service.send_message(msg)
    except Exception as te:
        print(f"Telegram Deployment Notify Error: {te}")
    
    return {"status": "success", "message": f"Strategy deployed successfully in {req.deployment_mode} mode", "deployment_id": doc["_id"]}

@app.post("/deploy/quick-nifty-atm")
async def deploy_quick_nifty_atm(mode: str = "MOCK"):
    return await deploy_quick_option_strategy(index="NIFTY", type="bullish", mode=mode)

@app.post("/deploy/quick-option-strategy")
async def deploy_quick_option_strategy(index: str = "NIFTY", type: str = "bullish", mode: str = "MOCK"):
    index = index.upper()
    if index == "NIFTY":
        index_key = "NSE_INDEX|Nifty 50"
        strike_step = 50
        lot_size = 25
    elif index == "SENSEX":
        index_key = "BSE_INDEX|SENSEX"
        strike_step = 100
        lot_size = 10
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported index: {index}. Only NIFTY and SENSEX are supported.")
        
    type = type.lower()
    if type not in ["bullish", "bearish"]:
        raise HTTPException(status_code=400, detail=f"Unsupported strategy type: {type}. Only bullish and bearish are supported.")

    # 1. Fetch Spot price
    spot = await upstox_service.get_spot_price(index_key)
    if not spot:
        raise HTTPException(status_code=400, detail=f"Could not fetch Spot price for {index}.")
        
    # 2. Expiries
    expiries = await upstox_service.get_expiry_dates(index_key)
    if not expiries:
        raise HTTPException(status_code=400, detail=f"Could not fetch option expiry dates for {index}.")
    nearest_expiry = expiries[0]
    
    # 3. Option Chain
    chain_res = await upstox_service.get_option_chain(index_key, nearest_expiry)
    chain = chain_res.get("chain", [])
    if not chain:
        raise HTTPException(status_code=400, detail=f"Could not fetch option chain for {index} expiry {nearest_expiry}.")
        
    atm_strike = round(spot / strike_step) * strike_step
    
    call_key = next((x['instrument_key'] for x in chain if x['strike_price'] == atm_strike and x['instrument_type'] == 'CE'), None)
    put_key = next((x['instrument_key'] for x in chain if x['strike_price'] == atm_strike and x['instrument_type'] == 'PE'), None)
    
    if not call_key or not put_key:
        raise HTTPException(status_code=400, detail=f"Could not find ATM options for strike {atm_strike}.")
        
    buy_oversold_strategy = {
        'id': 'buy_oversold',
        'name': 'Buy Oversold',
        'rules': [
             {'id': 'cond1', 'indicator': 'low_prev', 'operator': '<=', 'value': 'BBL_20_2_prev', 'valueType': 'indicator'},
             {'id': 'cond2', 'indicator': 'STOCHk_prev', 'operator': '<', 'value': '20', 'valueType': 'number'},
             {'id': 'cond3', 'indicator': 'STOCHk_prev', 'operator': '<=', 'value': 'STOCHd_prev', 'valueType': 'indicator'},
             {'id': 'cond4', 'indicator': 'STOCHk', 'operator': '>', 'value': 'STOCHd', 'valueType': 'indicator'},
             {'id': 'cond5', 'indicator': 'MACDh', 'operator': '>', 'value': 'MACDh_prev', 'valueType': 'indicator'}
        ]
    }

    sell_strategy = {
        'id': 'sell_strategy',
        'name': 'Sell Strategy',
        'rules': [
             {'id': 'srsi_overbought', 'indicator': 'STOCHRSIk_prev', 'operator': '>', 'value': '80', 'valueType': 'number'},
             {'id': 'srsi_cross_now', 'indicator': 'STOCHRSIk', 'operator': '<', 'value': 'STOCHRSId', 'valueType': 'indicator'},
             {'id': 'srsi_was_above', 'indicator': 'STOCHRSIk_prev', 'operator': '>=', 'value': 'STOCHRSId_prev', 'valueType': 'indicator'},
             {'id': 'macd_falling', 'indicator': 'MACDh', 'operator': '<', 'value': 'MACDh_prev', 'valueType': 'indicator'},
             {'id': 'macd_was_positive', 'indicator': 'MACDh_prev', 'operator': '>', 'value': '0', 'valueType': 'number'},
             {'id': 'red_candle', 'indicator': 'close', 'operator': '<', 'value': 'open', 'valueType': 'indicator'}
        ]
    }

    if type == "bullish":
        execution_plan = [
            {"id": 1, "leg": index_key, "strategyId": "buy_oversold", "timeframe": "5minute"},
            {"id": 2, "leg": call_key, "strategyId": "buy_oversold", "timeframe": "5minute"},
            {"id": 3, "leg": put_key, "strategyId": "sell_strategy", "timeframe": "5minute"}
        ]
        trade_instrument = call_key
    else:
        execution_plan = [
            {"id": 1, "leg": index_key, "strategyId": "sell_strategy", "timeframe": "5minute"},
            {"id": 2, "leg": call_key, "strategyId": "sell_strategy", "timeframe": "5minute"},
            {"id": 3, "leg": put_key, "strategyId": "buy_oversold", "timeframe": "5minute"}
        ]
        trade_instrument = put_key

    req = DeployStrategyRequest(
        instrument_key=None,
        interval="15minute",
        days_back=30,
        stop_loss=50.0,
        take_profit=100.0,
        is_advanced=True,
        execution_plan=execution_plan,
        saved_strategies=[buy_oversold_strategy, sell_strategy],
        trade_type="LONG",
        trade_instrument_key=trade_instrument,
        use_intraday=True,
        trailing_sl=True,
        trailing_sl_trigger_pct=40.0,
        deployment_mode=mode,
        quantity_type="MANUAL",
        quantity=1,
        lot_size=lot_size
    )
    
    return await deploy_strategy(req)

async def get_live_instrument_details(pk: str):
    """
    Fetches details of an instrument, checking local DB first,
    but dynamically resolving F&O instruments via live Upstox API if they are stale (expired).
    """
    if not pk:
        return {}
        
    info = await telegram_service.get_instrument_details(pk)
    
    # Check if F&O details are stale (expired)
    is_stale_fo = False
    expiry_str = info.get("expiry")
    if expiry_str and (pk.startswith("NSE_FO|") or pk.startswith("BSE_FO|")):
        try:
            from datetime import datetime as dt_class
            exp_date = dt_class.strptime(expiry_str, "%Y-%m-%d").date()
            if exp_date < dt_class.today().date():
                is_stale_fo = True
        except:
            try:
                import pandas as pd
                exp_date = pd.to_datetime(expiry_str).date()
                from datetime import date
                if exp_date < date.today():
                    is_stale_fo = True
            except:
                pass
                
    # If the option wasn't found in DB or is stale, try to dynamically resolve it via live Upstox API
    if (info.get("name") is None or is_stale_fo) and (pk.startswith("NSE_FO|") or pk.startswith("BSE_FO|")):
        try:
            import re
            quotes = upstox_service.get_market_quotes([pk])
            for qkey in quotes:
                if "|" in qkey:
                    parts = qkey.split("|")
                    if len(parts) == 2 and not parts[1].isdigit():
                        symbol = parts[1]
                        # 1. Weekly options format: e.g. NIFTY2662324100PE
                        match = re.search(r"^([A-Z]+)(\d{2})([1-9OND])(\d{2})(\d+)(CE|PE)$", symbol)
                        if match:
                            name, yy, m_char, dd, strike, opt_type = match.groups()
                            info["name"] = name
                            info["trading_symbol"] = symbol
                            info["strike"] = float(strike)
                            info["option_type"] = opt_type
                            
                            month_map = {'O': 10, 'N': 11, 'D': 12}
                            try:
                                m = month_map[m_char] if m_char in month_map else int(m_char)
                                from datetime import date as dt_date
                                info["expiry"] = dt_date(2000 + int(yy), m, int(dd)).strftime('%Y-%m-%d')
                            except Exception:
                                info["expiry"] = f"20{yy}-{m_char}-{dd}"
                            break
                        
                        # 2. Monthly options format: e.g. NIFTY26JUN24000CE
                        match = re.search(r"^([A-Z]+)(\d{2})([A-Z]{3})(\d+)(CE|PE)$", symbol)
                        if match:
                            name, yy, m_str, strike, opt_type = match.groups()
                            info["name"] = name
                            info["trading_symbol"] = symbol
                            info["strike"] = float(strike)
                            info["option_type"] = opt_type
                            
                            month_names = {
                                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                            }
                            m_str_upper = m_str.upper()
                            if m_str_upper in month_names:
                                m = month_names[m_str_upper]
                                try:
                                    year = 2000 + int(yy)
                                    import calendar
                                    from datetime import date as dt_date
                                    last_day = calendar.monthrange(year, m)[1]
                                    expiry_date = None
                                    for day in range(last_day, 0, -1):
                                        d = dt_date(year, m, day)
                                        if d.weekday() == 3: # Thursday
                                            expiry_date = d.strftime('%Y-%m-%d')
                                            break
                                    info["expiry"] = expiry_date or f"20{yy}-{m_str_upper}-28"
                                except Exception:
                                    info["expiry"] = f"20{yy}-{m_str_upper}"
                            else:
                                info["expiry"] = f"20{yy}-{m_str}"
                            break
        except Exception as e:
            print(f"Error dynamically resolving instrument {pk}: {e}")
            
    return info

def format_live_option_info(details: dict):
    """Formats options details as HTML text block for Telegram."""
    opt_type = details.get("option_type")
    if not opt_type:
        # Fallback check if it looks like an option symbol
        symbol = details.get("trading_symbol") or ""
        if symbol.endswith("CE") or symbol.endswith("PE"):
            opt_type = "CE" if symbol.endswith("CE") else "PE"
            import re
            match = re.search(r'(\d+)(?:CE|PE)$', symbol)
            strike = match.group(1) if match else "Unknown"
            name_match = re.search(r'^([A-Z]+)\d{2}[A-Z]{3}', symbol)
            name = name_match.group(1) if name_match else symbol
            
            opt_name = "Call (CE)" if opt_type == "CE" else "Put (PE)"
            return (
                f"<b>Name:</b> {name}\n"
                f"<b>Option Type:</b> {opt_name}\n"
                f"<b>Strike Price:</b> ₹{strike}\n"
            )
        return ""
        
    strike = details.get("strike")
    expiry = details.get("expiry")
    name = details.get("name")
    
    opt_name = "Call (CE)" if opt_type == "CE" else "Put (PE)"
    
    info = ""
    if name:
        info += f"<b>Name:</b> {name}\n"
    info += f"<b>Option Type:</b> {opt_name}\n"
    if strike is not None:
        try:
            strike_float = float(strike)
            strike_str = str(int(strike_float)) if strike_float.is_integer() else str(strike_float)
        except:
            strike_str = str(strike)
        info += f"<b>Strike Price:</b> ₹{strike_str}\n"
    if expiry:
        info += f"<b>Expiry:</b> {expiry}\n"
        
    return info

@app.get("/deploy/list")
async def list_deployments():
    cursor = app.mongodb["strategy_deployments"].find().sort("deployed_at", -1)
    docs = await cursor.to_list(length=100)
    result = []
    import re
    
    async def enrich_instrument(pk, d, prefix):
        if not pk: return
        info = await get_live_instrument_details(pk)
                        
        d[f"{prefix}name"] = info.get("name", "")
        # Fall back to display_name or pk if we still don't have a trading symbol
        d[f"{prefix}symbol"] = info.get("trading_symbol") or info.get("display_name") or pk
        d[f"{prefix}option_type"] = info.get("option_type", "")
        d[f"{prefix}strike"] = info.get("strike")
        d[f"{prefix}expiry"] = info.get("expiry", "")
        d[f"{prefix}type"] = info.get("instrument_type", "")

    for doc in docs:
        d = serialize_doc(doc)
        
        # Resolve primary_instrument details
        await enrich_instrument(d.get("primary_instrument"), d, "instrument_")
        
        # Resolve trade_instrument_key details
        await enrich_instrument(d.get("trade_instrument_key"), d, "trade_instrument_")
        
        # Resolve execution_plan legs details
        plan = d.get("execution_plan", [])
        if plan:
            for leg in plan:
                leg_key = leg.get("leg")
                if leg_key:
                    leg_details = {}
                    await enrich_instrument(leg_key, leg_details, "")
                    leg["instrument_details"] = leg_details
        
        result.append(d)
    
    return {"data": result}

@app.post("/deploy/stop/{deployment_id}")
async def stop_deployment(deployment_id: str):
    from bson import ObjectId
    await app.mongodb["strategy_deployments"].update_one(
        {"_id": ObjectId(deployment_id)},
        {"$set": {"status": "STOPPED", "stopped_at": datetime.now()}}
    )
    return {"status": "success"}

@app.post("/deploy/start/{deployment_id}")
async def start_deployment(deployment_id: str):
    from bson import ObjectId
    await app.mongodb["strategy_deployments"].update_one(
        {"_id": ObjectId(deployment_id)},
        {"$set": {"status": "ACTIVE", "resumed_at": datetime.now()}}
    )
    
    # Ensure engine is running
    if not trading_engine.active:
        await trading_engine.start_trading()
        
    return {"status": "success"}

@app.delete("/deploy/delete/{deployment_id}")
async def delete_deployment(deployment_id: str):
    from bson import ObjectId
    # Delete the deployment itself
    await app.mongodb["strategy_deployments"].delete_one({"_id": ObjectId(deployment_id)})
    # Delete all associated logs
    await app.mongodb["deployment_logs"].delete_many({"deployment_id": deployment_id})
    return {"status": "success"}

@app.post("/deploy/test/{deployment_id}")
async def test_deployment(deployment_id: str):
    from bson.objectid import ObjectId
    dep = await app.mongodb["strategy_deployments"].find_one({"_id": ObjectId(deployment_id)})
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")
        
    # Get LTP for primary instrument
    primary_instrument = dep.get("primary_instrument")
    ltp = 100.0
    try:
        quotes = upstox_service.get_market_quotes([primary_instrument])
        if primary_instrument in quotes:
            ltp = quotes[primary_instrument].get('ltp', 100.0)
    except: pass
    
    # Force log entry and trade execution
    print(f"Force testing deployment {deployment_id}")
    await app.mongodb["deployment_logs"].insert_one({
        "deployment_id": deployment_id,
        "timestamp": datetime.now(),
        "instrument": primary_instrument,
        "interval": dep.get('interval', 'N/A'),
        "close_price": ltp,
        "signal": True,
        "traded": True,
        "rules": [{"name": "TEST_TRIGGER", "result": True, "details": "Manual Override"}],
        "message": f"Manual Test Trigger initiated for {primary_instrument} at roughly {ltp}"
    })
    await trading_engine.execute_trade(dep, primary_instrument, ltp)
    return {"status": "success"}

@app.get("/deploy/logs/{deployment_id}")
async def get_deployment_logs(deployment_id: str):
    # Only fetch logs from the last 24 hours
    cutoff = datetime.now() - timedelta(hours=24)
    cursor = app.mongodb["deployment_logs"].find(
        {"deployment_id": deployment_id, "timestamp": {"$gte": cutoff}}
    ).sort("timestamp", -1).limit(500)
    docs = await cursor.to_list(length=500)
    return {"data": [serialize_doc(doc) for doc in docs]}

@app.get("/deploy/status")
async def get_engine_status():
    active_docs = await app.mongodb["strategy_deployments"].find({"status": "ACTIVE"}).to_list(100)
    return {
        "engine_active": trading_engine.active,
        "engine_has_db": trading_engine.mongodb is not None,
        "active_deployments_count": len(active_docs),
        "upstox_authenticated": upstox_service.access_token is not None,
    }

@app.get("/settings/telegram")
async def get_telegram_settings():
    config = await app.mongodb["settings"].find_one({"id": "telegram_config"})
    if config:
        config["_id"] = str(config["_id"])
        return config
    return {"bot_token": "", "chat_id": "", "group_name": "", "enabled": False}

@app.get("/settings/mirae")
async def get_mirae_settings():
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if config:
        config["_id"] = str(config["_id"])
        return config
    return {"mirae_api_key": "", "mirae_access_token": "", "enabled": False}

@app.post("/settings/mirae")
async def save_mirae_settings(req: Request):
    data = await req.json()
    await app.mongodb["settings"].update_one(
        {"id": "mirae_config"},
        {"$set": {
            "mirae_api_key": data.get("mirae_api_key"),
            "mirae_access_token": data.get("mirae_access_token"),
            "enabled": data.get("enabled", False),
            "updated_at": datetime.now()
        }},
        upsert=True
    )
    return {"status": "success"}

@app.get("/mirae/positions")
async def get_mirae_positions():
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if not config or not config.get("enabled"):
        return {"status": "error", "message": "Mirae Asset integration is disabled or missing credentials"}

    mirae_access_token = config.get("mirae_access_token") or config.get("access_token")
    api_key = config.get("mirae_api_key") or config.get("api_key")

    success, msg = mirae_service.initialize(mirae_access_token, api_key)
    if not success:
        return {"status": "error", "message": f"Login failed: {msg}"}

    # Fetch positions
    p_success, p_data = mirae_service.get_net_position()
    if not p_success:
        print("Mirae Positions Error:", p_data)
        return {"status": "error", "message": f"Failed to get positions: {p_data}"}

    print("Mirae Positions Data:", p_data)
    return {"status": "success", "data": p_data}

@app.get("/mirae/funds")
async def get_mirae_funds():
    config = await app.mongodb["settings"].find_one({"id": "mirae_config"})
    if not config or not config.get("enabled"):
        return {"status": "error", "message": "Mirae Asset integration is disabled"}

    mirae_access_token = config.get("mirae_access_token") or config.get("access_token")
    api_key = config.get("mirae_api_key") or config.get("api_key")
    
    success, msg = mirae_service.initialize(mirae_access_token, api_key)
    if not success:
        return {"status": "error", "message": f"Login failed: {msg}"}

    f_success, f_data = mirae_service.get_funds()
    if not f_success:
        print("Mirae Funds Error:", f_data)
        return {"status": "error", "message": f"Failed to get funds: {f_data}"}
        
    print("Mirae Funds Data:", f_data)
    return {"status": "success", "data": f_data}

@app.get("/settings/general")
async def get_general_settings():
    doc = await app.mongodb["settings"].find_one({"id": "general_config"})
    if doc:
        doc.pop("_id", None)
        return doc
    return {"enforce_market_hours": False, "pause_all_deployments": False}

@app.post("/settings/general")
async def save_general_settings(req: Request):
    data = await req.json()
    await app.mongodb["settings"].update_one(
        {"id": "general_config"},
        {"$set": {
            "enforce_market_hours": data.get("enforce_market_hours", False),
            "pause_all_deployments": data.get("pause_all_deployments", False),
            "updated_at": datetime.now()
        }},
        upsert=True
    )
    return {"status": "success"}

@app.post("/settings/general/toggle-pause")
async def toggle_pause_deployments():
    doc = await app.mongodb["settings"].find_one({"id": "general_config"})
    current = doc.get("pause_all_deployments", False) if doc else False
    new_val = not current
    await app.mongodb["settings"].update_one(
        {"id": "general_config"},
        {"$set": {"pause_all_deployments": new_val, "updated_at": datetime.now()}},
        upsert=True
    )
    return {"status": "success", "pause_all_deployments": new_val}

@app.post("/settings/telegram")
async def save_telegram_settings(req: Request):
    data = await req.json()
    await app.mongodb["settings"].update_one(
        {"id": "telegram_config"},
        {"$set": {
            "bot_token": data.get("bot_token"),
            "chat_id": data.get("chat_id"),
            "group_name": data.get("group_name"),
            "enabled": data.get("enabled", False),
            "updated_at": datetime.now()
        }},
        upsert=True
    )
    # Reload service config
    await telegram_service.load_config()
    return {"status": "success"}

@app.post("/settings/telegram/test")
async def test_telegram_settings(req: Request):
    data = await req.json()
    # Temporarily set config for test
    old_token = telegram_service.bot_token
    old_chat = telegram_service.chat_id
    old_enabled = telegram_service.enabled
    
    telegram_service.bot_token = data.get("bot_token")
    telegram_service.chat_id = data.get("chat_id")
    telegram_service.enabled = True
    
    msg = "🔔 <b>Test Message</b>\n\nYour RoboTrader Telegram integration is working perfectly!"
    success, err_msg = await telegram_service.send_message(msg)
    
    # Restore config
    telegram_service.bot_token = old_token
    telegram_service.chat_id = old_chat
    telegram_service.enabled = old_enabled
    
    if success:
        return {"status": "success"}
    else:
        return {"status": "error", "message": f"Telegram Error: {err_msg}"}

@app.post("/settings/telegram/detect")
async def detect_telegram_chat(req: Request):
    data = await req.json()
    # Temporarily set token for detection
    old_token = telegram_service.bot_token
    telegram_service.bot_token = data.get("bot_token")
    
    res, err_msg = await telegram_service.detect_chat_id()
    
    # Restore token
    telegram_service.bot_token = old_token
    
    if res:
        return {"status": "success", "data": res}
    else:
        return {"status": "error", "message": err_msg}


@app.post("/scanner/fetch-fno")
async def fetch_fno_from_nse():
    """
    Downloads FNO list from NSE website and updates fno_stocks collection
    """
    res = await scanner_populate.fetch_fno_list()
    return res

@app.post("/scanner/fetch-master")
async def fetch_master_instruments(exchange: str = "NSE"):
    """
    Downloads Upstox Master Instrument List (NSE/BSE) and populates upstox_collection
    """
    res = await scanner_populate.fetch_master_instruments(exchange=exchange)
    return res
