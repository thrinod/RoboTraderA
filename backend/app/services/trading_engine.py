import asyncio
import pandas as pd
from datetime import datetime
from .indicators import indicators
from .upstox_service import upstox_service

class TradingEngine:
    def __init__(self):
        self.active = False
        self.strategies = []
        self.mongodb = None # Will be set by main.py
        self.instrument_key = "NSE_INDEX|Nifty 50" # Default instrument

    def set_db(self, db):
        self.mongodb = db

    def start_trading(self):
        if self.active:
            print("Trading already active")
            return
        self.active = True
        print("Trading Engine Started")
        asyncio.create_task(self.run_loop())

    def stop_trading(self):
        self.active = False
        print("Trading Engine Stopped")

    async def run_loop(self):
        print("Starting Data Feeder Loop...")
        while self.active:
            try:
                await self.process_market_data()
            except Exception as e:
                print(f"Error in trading loop: {e}")
            
            await asyncio.sleep(5) # Fetch every 5 seconds

    async def process_market_data(self):
        # 1. Define time range (last 100 days/candles for calculation)
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = "2023-01-01" # Fetch robust history
        
        # 2. Fetch Data (Blocking call wrapped in thread if needed, here just direct for simplicity)
        # Note: In real app, run in executor.
        response = upstox_service.get_market_history(self.instrument_key, "1minute", from_date, to_date)
        
        if not response or not response.data or not response.data.candles:
            print("No data received from Upstox (check Token/Internet)")
            # Insert dummy update to show it's trying
            if self.mongodb is not None:
                await self.mongodb["upstox_collection"].insert_one({
                    "ticker": self.instrument_key,
                    "status": "Scanning (No Data from API)...",
                    "timestamp": datetime.now().isoformat()
                })
            return

        # 3. Convert to DataFrame
        candles = response.data.candles
        # Upstox returns [timestamp, open, high, low, close, volume, oi]
        df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        # Sort by timestamp ascending
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')

        # 4. Calculate Indicators
        df = indicators.calculate_all(df)
        
        # 5. Get Latest Candle with Indicators
        latest = df.iloc[-1].to_dict()
        
        # 6. Save to MongoDB
        record = {
            "ticker": self.instrument_key,
            "timestamp": latest['Timestamp'].isoformat(),
            "open": latest['Open'],
            "close": latest['Close'],
            "rsi": latest.get('RSI_14', 0), # Default to 0 if NaN
            "bb_upper": latest.get('BBU_5_2.0', 0), # Check actual col names from pandas-ta usually BBU_{len}_{std}
            "status": "Active Monitoring"
        }
        
        # Clean NaNs for MongoDB (it hates NaNs)
        for k, v in record.items():
            if pd.isna(v):
                record[k] = None

        if self.mongodb is not None:
            await self.mongodb["upstox_collection"].insert_one(record)
            print(f"Saved data for {self.instrument_key}: Close={latest['Close']}")

    def evaluate_market(self, ticker: str):
        pass # Moved logic to process_market_data

    def execute_trade(self, ticker, action):
        print(f"EXECUTING {action} on {ticker}")
        # upstox_service.place_order(...)

trading_engine = TradingEngine()
