import httpx
import logging

class TelegramService:
    def __init__(self):
        self.bot_token = None
        self.chat_id = None
        self.group_name = None
        self.enabled = False
        self.mongodb = None

    def set_db(self, db):
        self.mongodb = db

    async def get_symbol_info(self, instrument_key: str):
        """Helper to get a human readable symbol from instrument key using MongoDB"""
        if not instrument_key:
            return "Unknown"
        
        # Simple local caching to avoid database roundtrips for the same symbol
        if not hasattr(self, '_symbol_cache'):
            self._symbol_cache = {}
            
        if instrument_key in self._symbol_cache:
            return self._symbol_cache[instrument_key]
            
        if self.mongodb is not None:
            try:
                # Try scanner_instruments_main first
                instr = await self.mongodb["scanner_instruments_main"].find_one({"instrument_key": instrument_key})
                if not instr:
                    # Try upstox_collection
                    instr = await self.mongodb["upstox_collection"].find_one({"instrument_key": instrument_key})
                
                if instr:
                    symbol = instr.get('trading_symbol') or instr.get('name')
                    name = instr.get('name')
                    if symbol:
                        if name and name.upper() != symbol.upper():
                            display_name = f"{symbol} ({name})"
                        else:
                            display_name = symbol
                        self._symbol_cache[instrument_key] = display_name
                        return display_name
            except Exception as e:
                print(f"TelegramService: Error fetching symbol for {instrument_key}: {e}")
                
        # If not found or no DB, try to extract a clean representation from the key itself
        # e.g., "NSE_EQ|RELIANCE" -> "RELIANCE"
        # e.g., "NSE_FO|51349" -> "Expired/Unknown F&O (51349)"
        if "|" in instrument_key:
            parts = instrument_key.split("|")
            exchange = parts[0]
            token = parts[-1]
            if "FO" in exchange:
                return f"Expired/Unknown F&O ({token})"
            elif "EQ" in exchange or exchange in ["NSE", "BSE"]:
                return f"Expired/Unknown Equity ({token})"
            return token
            
        return instrument_key

    async def get_instrument_details(self, instrument_key: str, trading_symbol: str = None):
        """Helper to get full details of an instrument from MongoDB"""
        details = {
            "display_name": instrument_key,
            "trading_symbol": trading_symbol,
            "name": None,
            "is_option": False,
            "option_type": None,
            "strike": None,
            "expiry": None,
            "instrument_type": None,
        }
        
        if not instrument_key:
            return details
            
        if self.mongodb is not None:
            try:
                # Try scanner_instruments_main first
                instr = await self.mongodb["scanner_instruments_main"].find_one({"instrument_key": instrument_key})
                if not instr:
                    # Try upstox_collection
                    instr = await self.mongodb["upstox_collection"].find_one({"instrument_key": instrument_key})
                
                if instr:
                    symbol = instr.get('trading_symbol') or instr.get('name')
                    name = instr.get('name')
                    details["trading_symbol"] = symbol or trading_symbol
                    details["name"] = name
                    details["instrument_type"] = instr.get("instrument_type")
                    
                    if symbol:
                        if name and name.upper() != symbol.upper():
                            details["display_name"] = f"{symbol} ({name})"
                        else:
                            details["display_name"] = symbol
                            
                    # Option specific fields
                    opt_type = instr.get("option_type")
                    if opt_type:
                        details["is_option"] = True
                        details["option_type"] = opt_type.upper()
                    
                    strike = instr.get("strike")
                    if strike is not None:
                        details["strike"] = float(strike)
                        
                    expiry = instr.get("expiry")
                    if expiry:
                        details["expiry"] = expiry
                        
                    return details
            except Exception as e:
                print(f"TelegramService: Error fetching details for {instrument_key}: {e}")
                
        # Fallback / Parsing from key if not found in DB
        # E.g. key: "NSE_FO|50911"
        if "|" in instrument_key:
            parts = instrument_key.split("|")
            exchange = parts[0]
            token = parts[-1]
            if not details["trading_symbol"]:
                details["trading_symbol"] = token
            
            if "FO" in exchange:
                details["display_name"] = f"Expired/Unknown F&O ({token})"
            elif "EQ" in exchange or exchange in ["NSE", "BSE"]:
                details["display_name"] = f"Expired/Unknown Equity ({token})"
                
        return details

    async def get_option_display_info(self, instrument_key: str, trading_symbol: str = None):
        """Helper to return formatted HTML string of option details (CE/PE, strike, expiry) if the instrument is an option"""
        details = await self.get_instrument_details(instrument_key, trading_symbol)
        if not details.get("is_option"):
            # Check if trading symbol looks like an option (fallback for expired options)
            # E.g. "NATIONALUM26FEB355PE" or "NIFTY2652123500CE"
            symbol = details.get("trading_symbol") or ""
            if symbol.endswith("CE") or symbol.endswith("PE"):
                opt_type = "CE" if symbol.endswith("CE") else "PE"
                import re
                match = re.search(r'(\d+)(?:CE|PE)$', symbol)
                strike = match.group(1) if match else "Unknown"
                
                # Also try to extract a clean name prefix from the symbol
                # E.g. "NATIONALUM26FEB355PE" -> "NATIONALUM"
                name_match = re.search(r'^([A-Z]+)\d{2}[A-Z]{3}', symbol)
                name = name_match.group(1) if name_match else symbol
                
                opt_name = "Call (CE)" if opt_type == "CE" else "Put (PE)"
                return (
                    f"<b>Name:</b> {name}\n"
                    f"<b>Option Type:</b> {opt_name}\n"
                    f"<b>Strike Price:</b> ₹{strike}\n"
                )
            return ""
            
        opt_type = details.get("option_type")
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

    async def load_config(self):
        if self.mongodb is not None:
            config = await self.mongodb["settings"].find_one({"id": "telegram_config"})
            if config:
                self.bot_token = config.get("bot_token")
                self.chat_id = config.get("chat_id")
                self.group_name = config.get("group_name")
                self.enabled = config.get("enabled", False)
                return True
        return False

    async def send_message(self, message: str):
        if not self.enabled or not self.bot_token or not self.chat_id:
            return False, "Service not enabled or missing configuration"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=10.0)
                if response.status_code == 200:
                    return True, "Success"
                else:
                    error_data = response.json()
                    err_msg = error_data.get("description", "Unknown Telegram Error")
                    return False, err_msg
        except Exception as e:
            err_str = str(e)
            print(f"Telegram Error: {err_str}")
            return False, err_str

    async def detect_chat_id(self):
        if not self.bot_token:
            return None, "Bot Token missing"
            
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("result", [])
                    if not results:
                        return None, "No recent messages found. Please send a message to the group first!"
                    
                    # Search for the most recent GROUP or Channel update
                    # Iterate backwards to find the latest group message
                    for update in reversed(results):
                        chat = None
                        if "message" in update:
                            chat = update["message"]["chat"]
                        elif "my_chat_member" in update:
                            chat = update["my_chat_member"]["chat"]
                        elif "channel_post" in update:
                            chat = update["channel_post"]["chat"]
                            
                        if chat and chat.get("type") in ["group", "supergroup", "channel"]:
                            return {
                                "chat_id": str(chat["id"]),
                                "title": chat.get("title") or chat.get("username") or "Group Chat"
                            }, "Success (Detected Group)"
                    
                    # If no group found, take the very last update regardless of type
                    last_update = results[-1]
                    chat = None
                    if "message" in last_update:
                        chat = last_update["message"]["chat"]
                    
                    if chat:
                        return {
                            "chat_id": str(chat["id"]),
                            "title": chat.get("title") or chat.get("username") or "Private Chat"
                        }, "Success (No group found, detected last chat)"
                        
                    return None, "Could not extract chat info from updates."
                else:
                    return None, f"Telegram Error: {response.status_code}"
        except Exception as e:
            return None, str(e)

telegram_service = TelegramService()
