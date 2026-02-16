from datetime import datetime
import uuid
from .upstox_service import upstox_service

class MockTradeService:
    def __init__(self):
        self.db = None
        self.collection_name = "mock_trades"

    def set_db(self, db):
        self.db = db

    async def place_order(self, order_details):
        """
        Place a mock order.
        order_details: { instrument_key, quantity, transaction_type, price, product, etc }
        """
        try:
            if self.db is None:
                return {"status": "error", "message": "Database not connected"}

            # 1. Get Live Price for execution
            print(f"Mock Trade: Fetching quote for {order_details['instrument_key']}")
            quotes = upstox_service.get_market_quotes([order_details['instrument_key']])
            quote = quotes.get(order_details['instrument_key'])
            
            if not quote:
                 print(f"Mock Trade: Quote not found for {order_details['instrument_key']}")
                 return {"status": "error", "message": "Failed to fetch live price for execution"}

            avg_price = quote['ltp']
            
            trade_id = str(uuid.uuid4())
            
            trade_doc = {
                "trade_id": trade_id,
                "instrument_key": order_details['instrument_key'],
                "trading_symbol": quote.get('name') or order_details.get('trading_symbol') or  order_details['instrument_key'], # Fallback
                "quantity": int(order_details['quantity']),
                "transaction_type": order_details['transaction_type'], # BUY/SELL
                "product": "MOCK",
                "average_price": avg_price,
                "status": "OPEN",
                "timestamp": datetime.now().isoformat(),
                "pnl": 0.0,
                "history": [{
                    "action": "OPEN",
                    "price": avg_price,
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            print(f"Mock Trade: Inserting doc {trade_id}")
            await self.db[self.collection_name].insert_one(trade_doc)
            
            # Serialize _id for response
            if '_id' in trade_doc:
                trade_doc['_id'] = str(trade_doc['_id'])
                
            return {"status": "success", "message": "Mock Order Placed", "data": trade_doc}
        except Exception as e:
            print(f"Mock Trade Exception: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Server Error: {str(e)}"}

    async def get_positions(self):
        """
        Get all OPEN mock positions with real-time P&L.
        """
        if self.db is None:
            return []
            
        cursor = self.db[self.collection_name].find({"status": "OPEN"})
        positions = await cursor.to_list(length=100)
        
        if not positions:
            return []
            
        # Bulk Fetch Quotes
        keys = list(set([p['instrument_key'] for p in positions]))
        quotes = upstox_service.get_market_quotes(keys)
        
        results = []
        for p in positions:
            quote = quotes.get(p['instrument_key'])
            ltp = quote['ltp'] if quote else p['average_price']
            
            # Calculate P&L
            # For BUY: (LTP - Avg) * Qty
            # For SELL: (Avg - LTP) * Qty
            qty = p['quantity']
            buy_price = p['average_price']
            
            pnl = 0.0
            if p['transaction_type'] == 'BUY':
                pnl = (ltp - buy_price) * qty
            else:
                pnl = (buy_price - ltp) * qty
                
            # Add computed fields for frontend
            p['last_price'] = ltp
            p['pnl'] = round(pnl, 2)
            
            # Serialize _id
            if '_id' in p:
                p['_id'] = str(p['_id'])
                
            results.append(p)
            
        return results

    async def exit_position(self, trade_id):
        """
        Exit a mock position at market price.
        """
        try:
            if self.db is None:
                 return {"status": "error", "message": "Database not connected"}
                 
            trade = await self.db[self.collection_name].find_one({"trade_id": trade_id, "status": "OPEN"})
            if not trade:
                return {"status": "error", "message": "Position not found or already closed"}
                
            # Get Live Price
            quotes = upstox_service.get_market_quotes([trade['instrument_key']])
            quote = quotes.get(trade['instrument_key'])
            exit_price = quote['ltp'] if quote else trade['average_price']
            
            # Calculate Final P&L
            qty = trade['quantity']
            buy_price = trade['average_price']
            
            pnl = 0.0
            if trade['transaction_type'] == 'BUY':
                 pnl = (exit_price - buy_price) * qty
            else:
                 pnl = (buy_price - exit_price) * qty
                 
            # Update DB
            await self.db[self.collection_name].update_one(
                {"trade_id": trade_id},
                {
                    "$set": {
                        "status": "CLOSED",
                        "exit_price": exit_price,
                        "exit_timestamp": datetime.now().isoformat(),
                        "pnl": round(pnl, 2)
                    },
                    "$push": {
                        "history": {
                            "action": "EXIT",
                            "price": exit_price,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                }
            )
            
            return {"status": "success", "message": f"Exited at {exit_price}, P&L: {round(pnl, 2)}"}
        except Exception as e:
            print(f"Mock Exit Exception: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Exit Failed: {str(e)}"}

    async def get_history(self):
        """
        Get closed trades history.
        """
        if self.db is None:
            return []
        
        cursor = self.db[self.collection_name].find({"status": "CLOSED"}).sort("exit_timestamp", -1)
        trades = await cursor.to_list(length=500)
        
        # Convert _id to str
        for t in trades:
            t['_id'] = str(t['_id'])
            
        return trades

mock_trade_service = MockTradeService()
