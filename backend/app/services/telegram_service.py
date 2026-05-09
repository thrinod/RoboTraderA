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
