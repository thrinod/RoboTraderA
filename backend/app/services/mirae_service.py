import logging
import sys
import traceback
from tradingapi_a.mconnect import MConnect
from datetime import datetime

class MiraeService:
    def __init__(self):
        self.mconnect = None
        self.logger = logging.getLogger(__name__)

    def initialize(self, access_token, api_key=None):
        try:
            self.mconnect = MConnect()
            self.logger.info("MConnect object created")
            if api_key:
                self.mconnect.set_api_key(api_key)
            self.mconnect.set_access_token(access_token)
            self.logger.info("Mirae Access Token set successfully")
            return True, "Initialized"
        except Exception as e:
            type_, value_, traceback_ = sys.exc_info()
            stack_trace = traceback.format_exception(type_, value_, traceback_)
            self.logger.error(stack_trace)
            return False, str(e)

    def get_net_position(self):
        if not self.mconnect:
            return False, "Mirae service not initialized"
        try:
            res = self.mconnect.get_net_position()
            return True, res.json()
        except Exception as e:
            return False, str(e)

    def cancel_all_orders(self):
        if not self.mconnect:
            return False, "Mirae service not initialized"
        try:
            res = self.mconnect.cancel_all()
            return True, res.json()
        except Exception as e:
            return False, str(e)

    def get_funds(self):
        if not self.mconnect:
            return False, "Mirae service not initialized"
        try:
            res = self.mconnect.get_fund_summary()
            return True, res.json()
        except Exception as e:
            return False, str(e)

    # Note: Need more methods based on actual usage, like convert_position or square_off
    # We will need the exact parameters to exit a position for Mirae.

mirae_service = MiraeService()
