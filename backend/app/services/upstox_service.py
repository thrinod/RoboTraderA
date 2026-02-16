import os
import httpx
from upstox_client.api_client import ApiClient
from upstox_client.api.login_api import LoginApi
from upstox_client.api.history_api import HistoryApi
from upstox_client.api.order_api import OrderApi
from upstox_client.api.market_quote_api import MarketQuoteApi
from upstox_client.models.place_order_request import PlaceOrderRequest
import upstox_client.configuration as config
from typing import Optional
import pandas as pd
import pandas_ta as ta
import datetime
from urllib.parse import quote
import asyncio


class UpstoxService:
    def __init__(self):
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.client_id = os.getenv("UPSTOX_API_KEY") # Upstox uses API Key as Client ID often
        self.client_secret = os.getenv("UPSTOX_API_SECRET")
        self.redirect_uri = os.getenv("UPSTOX_REDIRECT_URI")
        self.base_url = "https://api.upstox.com/v2"
        self.access_token: Optional[str] = None
        
        # Configuration for Upstox Client
        self.configuration = config.Configuration()
        # Ensure we set the access token when available
        self.db = None
        self.expiry_cache = {} # { 'instrument_key': { 'timestamp': time, 'data': [] } }

        

    def verify_token(self, token: str) -> bool:
        """
        Verifies the token by making a dummy call to Upstox API (e.g. User Profile).
        """
        self.configuration.access_token = token
        try:
            api_client = ApiClient(self.configuration)
            from upstox_client.api.user_api import UserApi
            user_api = UserApi(api_client)
            response = user_api.get_profile("2.0")
            print(f"Token Verified. User: {response.data.user_name}")
            return True
        except Exception as e:
            print(f"Token Verification Failed: {e}")
            return False

    def has_valid_token(self) -> bool:
        return self.access_token is not None

    def _clean_keys(self, data):
        if isinstance(data, dict):
            return {k.lstrip('_'): self._clean_keys(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_keys(i) for i in data]
        return data

    def get_funds(self):
        try:
            api_client = ApiClient(self.configuration)
            from upstox_client.api.user_api import UserApi
            user_api = UserApi(api_client)
            response = user_api.get_user_fund_margin("2.0")
            # Convert to dict and clean keys
            data = response.data.to_dict() if hasattr(response.data, 'to_dict') else response.data
            return self._clean_keys(data)
        except Exception as e:
            print(f"Error fetching funds: {e}")
            return None

    def get_positions(self):
        try:
            api_client = ApiClient(self.configuration)
            from upstox_client.api.portfolio_api import PortfolioApi
            portfolio_api = PortfolioApi(api_client)
            response = portfolio_api.get_positions("2.0")
            # Convert list of objects to list of dicts
            data = [p.to_dict() for p in response.data] if response and response.data else []
            return {'data': self._clean_keys(data)}
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return None

    
    async def get_expiry_dates(self, instrument_key: str):
        """
        Fetches available expiry dates for an index.
        """
        try:
            url = "https://api.upstox.com/v2/option/contract"
            params = {"instrument_key": instrument_key}
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.access_token}"
            }
            import httpx
            import time
            
            # 1. Check Cache (TTL 1 hour)
            if instrument_key in self.expiry_cache:
                cached = self.expiry_cache[instrument_key]
                if (time.time() - cached['timestamp']) < 3600:
                    print(f"Returning Cached Expiries for {instrument_key}")
                    return cached['data']

            start_time = time.time()
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
                t1 = time.time()
                print(f"Expiry Query Time: {t1 - start_time:.2f}s | Status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"Option Contract API Error: {response.text}")
                    return ["2025-12-28", "2026-01-04", "2026-01-11"]
                
                t2 = time.time()
                data = response.json()
                t3 = time.time()
                print(f"Expiry JSON Parse Time: {t3 - t2:.2f}s")
                
                if 'data' in data:
                    raw_len = len(data['data'])
                    # Extract unique expiry dates
                    expiries = set()
                    for contract in data['data']:
                        if 'expiry' in contract:
                            expiries.add(contract['expiry'])
                    
                    sorted_exp = sorted(list(expiries))
                    t4 = time.time()
                    print(f"Expiry Extraction Time: {t4 - t3:.2f}s | Raw Contracts: {raw_len} | Unique Expiries: {len(sorted_exp)}")
                    
                    # 2. Set Cache
                    self.expiry_cache[instrument_key] = {
                        'timestamp': time.time(),
                        'data': sorted_exp
                    }
                    return sorted_exp
                return []
        except Exception as e:
            print(f"Error fetching expiries: {e}")
            return ["2025-12-28"] # Demo fallback

    async def get_spot_price(self, instrument_key: str):
        """
        Fetches the latest LTP for the underlying index (Spot Price).
        """
        try:
            url = "https://api.upstox.com/v2/market-quote/ltp"
            params = {"instrument_key": instrument_key}
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.access_token}"
            }
            
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
                # DEBUG PRINT
                print(f"Spot Price API Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    # DEBUG PRINT
                    print(f"Spot Price Response Keys: {data.get('data', {}).keys()}")
                    
                    # Response format: { "status": "success", "data": { "NSE_INDEX|Nifty 50": { "last_price": 21456.0, ... } } }
                    if 'data' in data:
                        # Try exact match
                        if instrument_key in data['data']:
                            price = data['data'][instrument_key]['last_price']
                            print(f"Spot Price Found (Exact): {price}")
                            return price
                        
                        # Fallback: Return the first item's price (since we requested only one)
                        for key, val in data['data'].items():
                             if 'last_price' in val:
                                 price = val['last_price']
                                 print(f"Spot Price Found (Fallback from {key}): {price}")
                                 return price
                                 
                        print(f"Instrument Key {instrument_key} not found in response.")
            return None
        except Exception as e:
            print(f"Error fetching spot price: {e}")
            return None

    async def get_option_chain(self, instrument_key: str, expiry_date: str):
        """
        Fetches option chain data.
        instrument_key: e.g. 'NSE_INDEX|Nifty 50'
        expiry_date: '2023-12-28' (ISO format)
        """
        try:
            url = "https://api.upstox.com/v2/option/chain"
            params = {
                "instrument_key": instrument_key,
                "expiry_date": expiry_date
            }
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.access_token}"
            }
            
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
                
                if response.status_code != 200:
                    print(f"Option Chain API Error: {response.text}")
                    return []
                    
                data = response.json()
                if 'data' in data:
                    # Transform Upstox V2 (Assuming grouped by strike) to Flat List for Frontend
                    # Structure: [{ "strike_price": ..., "call_options": { "market_data": {...} }, "put_options": {...} }]
                    flat_data = []
                    raw_list = data['data']
                    
                    total_ce_oi = 0
                    total_pe_oi = 0

                    for item in raw_list:
                        strike = item.get('strike_price')
                        
                        # Process CE
                        if 'call_options' in item and item['call_options']:
                            ce = item['call_options']
                            md = ce.get('market_data', {})
                            greeks = ce.get('option_greeks', {})
                            
                            ltp = md.get('ltp', 0)
                            oi = md.get('oi', 0)
                            close = md.get('close_price', 0)
                            change = round(ltp - close, 2) if close > 0 else 0
                            delta = greeks.get('delta', 0)
                            oi_value = oi * ltp
                            
                            total_ce_oi += oi

                            ce_record = {
                                'strike_price': strike,
                                'instrument_type': 'CE',
                                'instrument_key': ce.get('instrument_key'),
                                'last_price': ltp,
                                'open_interest': oi,
                                'volume': md.get('volume', 0),
                                'price_change': change,
                                'bid_price': md.get('bid_price', 0),
                                'ask_price': md.get('ask_price', 0),
                                'iv': greeks.get('iv', 0),
                                'delta': delta,
                                'oi_value': oi_value,
                                'lot_size': ce.get('lot_size', 0)
                            }
                            flat_data.append(ce_record)
                            
                        # Process PE
                        if 'put_options' in item and item['put_options']:
                            pe = item['put_options']
                            md = pe.get('market_data', {})
                            greeks = pe.get('option_greeks', {})
                            
                            ltp = md.get('ltp', 0)
                            oi = md.get('oi', 0)
                            close = md.get('close_price', 0)
                            change = round(ltp - close, 2) if close > 0 else 0
                            delta = greeks.get('delta', 0)
                            oi_value = oi * ltp
                            
                            total_pe_oi += oi

                            pe_record = {
                                'strike_price': strike,
                                'instrument_type': 'PE',
                                'instrument_key': pe.get('instrument_key'),
                                'last_price': ltp,
                                'open_interest': oi,
                                'volume': md.get('volume', 0),
                                'price_change': change,
                                'bid_price': md.get('bid_price', 0),
                                'ask_price': md.get('ask_price', 0),
                                'iv': greeks.get('iv', 0),
                                'delta': delta,
                                'oi_value': oi_value,
                                'lot_size': pe.get('lot_size', 0)
                            }
                            flat_data.append(pe_record)

                    return { 'chain': flat_data, 'totals': { 'ce': total_ce_oi, 'pe': total_pe_oi } }
                return { 'chain': [], 'totals': { 'ce': 0, 'pe': 0 } }
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            return { 'chain': [], 'totals': { 'ce': 0, 'pe': 0 } }






    async def calculate_indicators(self, df, instrument_key=None, limit=100, pivot_data=None):
        """
        Calculates Technical Indicators and Pivots.
        """
        try:
            # --- INDICATOR CALCULATIONS ---
            # 1. RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # 2. MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            
            # 3. SMA
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)

            # 4. Stochastic RSI
            stoch_rsi = ta.stochrsi(df['close'], length=14, rsi_length=14, k=3, d=3)
            
            # 5. ADX / DMI
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)

            # 6. Bollinger Bands
            bb_data = None
            bb = ta.bbands(df['close'], length=20, std=2)
            if bb is not None:
                    bb.columns = ['bb_lower', 'bb_middle', 'bb_upper', 'bb_bandwidth', 'bb_percent']
                    df = pd.concat([df, bb], axis=1)
                    bb_data = bb
            
            # 7. PIVOT POINTS
            # Note: pivot_data arg allows passing pre-fetched data to avoid sequential fetches loop
            if pivot_data is None and instrument_key:
                pivot_data = await self._fetch_daily_pivot_data_async(instrument_key)

            # Safe Fallback for Pivots
            if pivot_data is None and not df.empty:
                 try:
                    # Fallback: Use First Candle of Intraday Data
                    first = df.iloc[0]
                    # Approx Yesterday Close ~ Open of First
                    fp = (first['high'] + first['low'] + first['open']) / 3
                    pivot_data = {
                        'pivot': round(fp, 2),
                        'r1': round((2 * fp) - first['low'], 2),
                        's1': round((2 * fp) - first['high'], 2),
                        'r2': round(fp + (first['high'] - first['low']), 2),
                        's2': round(fp - (first['high'] - first['low']), 2),
                        'r3': round(first['high'] + 2 * (fp - first['low']), 2),
                        's3': round(first['low'] - 2 * (first['high'] - fp), 2)
                    }
                 except: pass

            return self._calculate_indicators_sync(df, macd, stoch_rsi, adx_data, bb, pivot_data, limit)

        except Exception as e:
            print(f"Calculation Error: {e}")
            return {}, {}, {}

    def _calculate_indicators_sync(self, df, macd, stoch_rsi, adx_data, bb, pivot_data, limit):
        import pandas as pd
        # Helper Helpers
        def get_series(series):
            if series is None: return []
            return series.where(pd.notnull(series), None).tolist()

        def get_series_df(df_in, col_idx=0, col_name=None, subset_len=100):
            if df_in is None or len(df_in) == 0: return []
            try:
                sliced = df_in.iloc[-subset_len:]
                if col_name and col_name in sliced.columns:
                    return get_series(sliced[col_name])
                elif col_idx < len(sliced.columns):
                    return get_series(sliced.iloc[:, col_idx])
                return []
            except: return []

        # Subset for Series
        subset = df.tail(limit)
        
        indicators_series = {
            'rsi': get_series(subset['rsi']),
            'stoch_k': get_series_df(stoch_rsi, col_idx=0, subset_len=limit), 
            'stoch_d': get_series_df(stoch_rsi, col_idx=1, subset_len=limit),
            'macd': get_series_df(macd, col_name='MACD_12_26_9', subset_len=limit),
            'macd_hist': get_series_df(macd, col_name='MACDh_12_26_9', subset_len=limit),
            'macd_signal': get_series_df(macd, col_name='MACDs_12_26_9', subset_len=limit),
            'adx': get_series_df(adx_data, col_name='ADX_14', subset_len=limit),
            'dmp': get_series_df(adx_data, col_name='DMP_14', subset_len=limit),
            'dmn': get_series_df(adx_data, col_name='DMN_14', subset_len=limit),
            'sma_50': get_series(subset['sma_50']),
            'sma_200': get_series(subset['sma_200']),
            'bb_upper': get_series(subset['bb_upper']) if 'bb_upper' in subset.columns else [],
            'bb_middle': get_series(subset['bb_middle']) if 'bb_middle' in subset.columns else [],
            'bb_lower': get_series(subset['bb_lower']) if 'bb_lower' in subset.columns else []
        }

        # Latest Values
        last_row = df.iloc[-1]
        
        def get_val_named(df_in, col_name):
            if df_in is None or col_name not in df_in.columns or len(df_in) == 0: return 0
            val = df_in.iloc[-1][col_name]
            return val if not pd.isna(val) else 0

        def get_val_idx(df_in, col_idx):
            if df_in is None or len(df_in) == 0: return 0
            val = df_in.iloc[-1, col_idx]
            return val if not pd.isna(val) else 0

        indicators = {
            'rsi': last_row['rsi'] if not pd.isna(last_row['rsi']) else 0,
            'macd': macd['MACD_12_26_9'].iloc[-1] if not pd.isna(macd['MACD_12_26_9'].iloc[-1]) else 0,
            'macd_hist': macd['MACDh_12_26_9'].iloc[-1] if not pd.isna(macd['MACDh_12_26_9'].iloc[-1]) else 0,
            'macd_signal': macd['MACDs_12_26_9'].iloc[-1] if not pd.isna(macd['MACDs_12_26_9'].iloc[-1]) else 0,
            'sma_50': last_row['sma_50'] if not pd.isna(last_row['sma_50']) else 0,
            'sma_200': last_row['sma_200'] if not pd.isna(last_row['sma_200']) else 0,
            'stoch_k': get_val_idx(stoch_rsi, 0),
            'stoch_d': get_val_idx(stoch_rsi, 1),
            'adx': get_val_named(adx_data, 'ADX_14'),
            'dmp': get_val_named(adx_data, 'DMP_14'), 
            'dmn': get_val_named(adx_data, 'DMN_14'),
            'pivot_points': pivot_data,
            'bb_upper': get_val_named(bb, 'bb_upper') if bb is not None else 0,
            'bb_middle': get_val_named(bb, 'bb_middle') if bb is not None else 0,
            'bb_lower': get_val_named(bb, 'bb_lower') if bb is not None else 0
        }
        
        return indicators, indicators_series, pivot_data
    async def _fetch_historical_df(self, instrument_key: str, interval: str, days_back_override: int = None):
        """
        Helper to fetch Historical Dataframe with Resampling logic.
        """
        try:
             # Determine Upstox-supported fetch interval and local resampling rule
            # Upstox supports: 1minute, 30minute, day, week, month
            fetch_interval = "1minute" # Default
            resample_rule = None
            days_back = 5

            if interval == "day":
                fetch_interval = "day"
                days_back = 30 # User requested 30 days
                resample_rule = None
            elif interval == "1minute":
                fetch_interval = "1minute"
                days_back = 30 # Reverted to 30 as per user request (removed optimization)
                resample_rule = None
            elif interval == "3minute":
                fetch_interval = "1minute"
                days_back = 30 # Reverted
                resample_rule = "3T"
            elif interval == "5minute":
                fetch_interval = "1minute"
                days_back = 30 # Reverted
                resample_rule = "5T"
            elif interval == "15minute":
                fetch_interval = "1minute"
                 # Optimization: specific 15min api? No, stick to 1min resampling for consistency
                days_back = 45 # Enough for resampling
                resample_rule = "15T"
            elif interval == "30minute":
                fetch_interval = "30minute"
                days_back = 60
                resample_rule = None
            elif interval == "60minute":
                # 60min is not native, use 30min for efficiency
                fetch_interval = "30minute" 
                days_back = 60 # Need ~70 days for 200 SMA (approx 300 hours), giving buffer
                resample_rule = "60T"
            else:
                # Fallback
                fetch_interval = interval
                days_back = 30 # Default reverted
                resample_rule = None 
            
            if days_back_override:
                days_back = days_back_override

            # Fetch Direct from API
            import datetime
            from urllib.parse import quote
            import pandas as pd
            import httpx

            today = datetime.date.today()
            # Creating a buffer of 1 day into the future to ensure today's data is captured.
            to_date = today + datetime.timedelta(days=1)
            from_date = today - datetime.timedelta(days=days_back)
            
            to_date_str = to_date.strftime('%Y-%m-%d')
            from_date_str = from_date.strftime('%Y-%m-%d')

            encoded_key = quote(instrument_key)
            if interval == "day":
                 url = f"{self.base_url}/historical-candle/{encoded_key}/day/{to_date_str}/{from_date_str}"
            else:
                 url = f"{self.base_url}/historical-candle/{encoded_key}/{fetch_interval}/{to_date_str}/{from_date_str}"
            
            # print(f"DEBUG: Fetching Raw History from: {url}")
            headers = {"accept": "application/json"}
            if self.access_token:
                 headers["Authorization"] = f"Bearer {self.access_token}"
            
            data = None
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                data = resp.json()
            
            if data and data.get('status') == 'success' and data.get('data', {}).get('candles'):
                candles = data['data']['candles']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Standardize Columns
                cols = ['open', 'high', 'low', 'close', 'volume']
                for col in cols:
                    df[col] = pd.to_numeric(df[col])

                # Resample if needed
                if resample_rule:
                    conversion = {
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }
                    df = df.resample(resample_rule).agg(conversion).dropna()
                
                return df
            else:
                 return pd.DataFrame() # Empty
        except Exception as e:
            print(f"Fetch History Error for {instrument_key}: {e}")
            return pd.DataFrame()

    async def _fetch_daily_pivot_data_async(self, instrument_key: str):
        """
        Helper to fetch daily data for pivot calculation.
        """
        try:
            import datetime
            from urllib.parse import quote
            
            # Calculate dates for Daily Fetch (Last 5 days)
            today_date = datetime.date.today()
            p_end = today_date + datetime.timedelta(days=1)
            p_start = today_date - datetime.timedelta(days=5)
            p_url = f"https://api.upstox.com/v3/historical-candle/{quote(instrument_key)}/day/{p_end.strftime('%Y-%m-%d')}/{p_start.strftime('%Y-%m-%d')}"
            
            # Fetch daily data
            async with httpx.AsyncClient(timeout=10.0) as p_client:
                p_headers = {"accept": "application/json"}
                if self.access_token: p_headers["Authorization"] = f"Bearer {self.access_token}"
                
                p_resp = await p_client.get(p_url, headers=p_headers)
                p_json = p_resp.json()
                
                if p_json and p_json.get('status') == 'success' and p_json.get('data', {}).get('candles'):
                    daily_candles = p_json['data']['candles']
                    df_d = pd.DataFrame(daily_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    
                    # Sort by date
                    df_d['timestamp'] = pd.to_datetime(df_d['timestamp'])
                    if df_d['timestamp'].dt.tz is not None:
                        df_d['timestamp'] = df_d['timestamp'].dt.tz_localize(None)
                    
                    df_d.sort_values('timestamp', inplace=True)
                    
                    # We need at least 1 closed candle before today.
                    today_ts = pd.Timestamp(datetime.date.today())
                    
                    closed_days = df_d[df_d['timestamp'] < today_ts]
                    
                    if len(closed_days) > 0:
                        yesterday = closed_days.iloc[-1]
                        yp_high = float(yesterday['high'])
                        yp_low = float(yesterday['low'])
                        yp_close = float(yesterday['close'])
                        
                        pivot = (yp_high + yp_low + yp_close) / 3
                        r1 = (2 * pivot) - yp_low
                        s1 = (2 * pivot) - yp_high
                        r2 = pivot + (yp_high - yp_low)
                        s2 = pivot - (yp_high - yp_low)
                        r3 = yp_high + 2 * (pivot - yp_low)
                        s3 = yp_low - 2 * (yp_high - pivot)
                        
                        pivots = {
                            'pivot': round(pivot, 2),
                            'r1': round(r1, 2),
                            's1': round(s1, 2),
                            'r2': round(r2, 2),
                            's2': round(s2, 2),
                            'r3': round(r3, 2),
                            's3': round(s3, 2)
                        }
                        
                        return pivots
            return None
        except Exception as e:
            print(f"Pivot Async Fetch Error for {instrument_key}: {e}")
            return None

    async def _fetch_intraday_data_raw(self, instrument_key: str, interval: str):
        """
        Helper to fetch raw intraday data from Upstox API.
        """
        try:
            # Parse Interval
            unit = "minutes"
            in_val = "1"
            
            if "minute" in interval:
                 unit = "minutes"
                 in_val = interval.replace("minute", "")
            elif interval == "day":
                 unit = "days"
                 in_val = "1"
            
            path_interval = f"{unit}/{in_val}"
            encoded_key = quote(instrument_key)
            url = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded_key}/{path_interval}"
            
            headers = {"accept": "application/json"}
            if self.access_token:
                 headers["Authorization"] = f"Bearer {self.access_token}"
            
            intraday_df = pd.DataFrame()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                     data = response.json()
                     if data.get("status") == "success":
                         candles = data.get("data", {}).get("candles", [])
                         formatted = []
                         for c in candles:
                             formatted.append({
                                 "timestamp": c[0],
                                 "open": c[1],
                                 "high": c[2],
                                 "low": c[3],
                                 "close": c[4],
                                 "volume": c[5],
                                 "oi": c[6]
                             })
                         intraday_df = pd.DataFrame(formatted)
            return intraday_df
        except Exception as e:
            print(f"Intraday Direct Fetch Error for {instrument_key}: {e}")
            return pd.DataFrame()

    async def get_intraday_candles(self, instrument_key: str, interval: str = "1minute"):
        """
        Fetch intraday candles + recent history to ensure full indicator visibility.
        """
        try:
            # Sequential Fetching (User Requested)
            # 1. History
            # 2. Intraday
            # 3. Pivot Data (Daily)
            
            history_df = await self._fetch_historical_df(instrument_key, interval)
            intraday_df = await self._fetch_intraday_data_raw(instrument_key, interval)
            pivot_data_result = await self._fetch_daily_pivot_data_async(instrument_key)

            # Process Intraday DF
            if not intraday_df.empty:
                cols = ['open', 'high', 'low', 'close', 'volume']
                for col in cols:
                    intraday_df[col] = pd.to_numeric(intraday_df[col])
                
                intraday_df['timestamp'] = pd.to_datetime(intraday_df['timestamp'])
                if intraday_df['timestamp'].dt.tz is None:
                    # Intraday usually comes with offset, but if not
                    intraday_df['timestamp'] = intraday_df['timestamp'].dt.tz_localize('Asia/Kolkata')
                
                # Ensure it's timezone aware matching history (if history is tz-naive, we adjust)
                if not history_df.empty and history_df.index.tz is None:
                     # History DF from _fetch is timestamp indexed and likely tz-naive or UTC?
                     # Standardize to tz-naive for simplicity in merging
                     intraday_df['timestamp'] = intraday_df['timestamp'].dt.tz_localize(None)
                
                intraday_df.set_index('timestamp', inplace=True)
                intraday_df.sort_index(inplace=True)

            # Merge
            full_df = pd.DataFrame()
            if not history_df.empty and not intraday_df.empty:
                # Combine and drop duplicates (keep last/intraday version if overlap?)
                full_df = pd.concat([history_df, intraday_df])
                full_df = full_df[~full_df.index.duplicated(keep='last')]
            elif not history_df.empty:
                full_df = history_df
            elif not intraday_df.empty:
                full_df = intraday_df
            
            if full_df.empty:
                return None
            
            full_df.sort_index(inplace=True)
            
            # Calculate Indicators
            # Limit series to last 500 points (approx 1.5 days) for display, 
            # but calculation uses full history automatically via pandas_ta inside the function? 
            # No, calculate_indicators takes 'limit' which subsets the SERIES returned.
            # The calculation itself (ta.sma, etc) runs on the whole 'df' passed.
            # So passing full_df ensures good values. limit=500 returns 500 points.
            limit_candles = 500
            indicators, series, pivots = await self.calculate_indicators(full_df, instrument_key=instrument_key, limit=limit_candles, pivot_data=pivot_data_result)
            
            # Slice candles to match limit
            display_df = full_df.tail(limit_candles)

            def clean_for_json(obj):
                import math
                if isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                    return obj
                elif isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(i) for i in obj]
                return obj

            result = {
                "candles": display_df.reset_index().assign(timestamp=lambda x: x['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S+05:30')).to_dict(orient="records"),
                "indicators": indicators, # Latest values
                "indicators_series": series, # Timelines (matched to limit)
                "ltp": display_df.iloc[-1]["close"],
                "change": display_df.iloc[-1]["close"] - display_df.iloc[-2]["close"] if len(display_df) > 1 else 0
            }
            return clean_for_json(result)

        except Exception as e:
            print(f"Error in get_intraday_candles: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "trace": traceback.format_exc()}


    async def get_instrument_history(self, instrument_key: str, interval="1minute"):
        """
        Fetches historical data for ANY instrument and calculates indicators.
        """
        try:
            # Determine Upstox-supported fetch interval and local resampling rule
            # Upstox supports: 1minute, 30minute, day, week, month
            fetch_interval = "1minute" # Default
            resample_rule = None
            days_back = 5

            if interval == "day":
                fetch_interval = "day"
                days_back = 600
                resample_rule = None
            elif interval == "1minute":
                fetch_interval = "1minute"
                days_back = 30
                resample_rule = None
            elif interval == "3minute":
                fetch_interval = "1minute"
                days_back = 30
                resample_rule = "3T"
            elif interval == "5minute":
                fetch_interval = "1minute"
                days_back = 30
                resample_rule = "5T"
            elif interval == "15minute":
                fetch_interval = "1minute"
                 # Optimization: specific 15min api? No, stick to 1min resampling for consistency
                days_back = 45 
                resample_rule = "15T"
            elif interval == "30minute":
                fetch_interval = "30minute"
                days_back = 100
                resample_rule = None
            elif interval == "60minute":
                # 60min is not native, use 30min for efficiency
                fetch_interval = "30minute" 
                days_back = 200 # Need ~70 days for 200 SMA (approx 300 hours), giving buffer
                resample_rule = "60T"
            else:
                # Fallback
                fetch_interval = interval
                days_back = 30
                resample_rule = None 

            # Fetch Direct from API
            today = datetime.date.today()
            # Creating a buffer of 1 day into the future to ensure today's data is captured.
            tomorrow = today + datetime.timedelta(days=1)
            
            # API Call Logic using Historical API (Standard V3)
            # URL: https://api.upstox.com/v2/historical-candle/{instrumentKey}/{interval}/{to_date}/{from_date}
            
            # Duration logic
            days_back = 5
            if "minute" in interval:
                days_back = 30 
            elif "day" in interval:
                days_back = 365
            
            to_date = datetime.date.today() + datetime.timedelta(days=1)
            from_date = datetime.date.today() - datetime.timedelta(days=days_back)
            
            to_date_str = to_date.strftime('%Y-%m-%d')
            from_date_str = from_date.strftime('%Y-%m-%d')

            encoded_key = quote(instrument_key)
            if interval == "day":
                 url = f"{self.base_url}/historical-candle/{encoded_key}/day/{to_date_str}/{from_date_str}"
            else:
                 # Default to 1minute mapping if needed, or pass interval directly if compatible
                 # V2 API expected '1minute', '30minute' etc.
                 url = f"{self.base_url}/historical-candle/{encoded_key}/{fetch_interval}/{to_date_str}/{from_date_str}"
            
            print(f"Fetching history from: {url}")



            
            # Use minimal headers for V3 (Bearer only)
            headers = {"accept": "application/json"}
            if self.access_token:
                 headers["Authorization"] = f"Bearer {self.access_token}"
            
            data = None
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    print(f"DEBUG: Fetching URL: {url}")
                    resp = await client.get(url, headers=headers)
                    print(f"DEBUG: Response Status: {resp.status_code}")
                    data = resp.json()
                    # print(f"DEBUG: Response Keys: {data.keys() if data else 'None'}")
            except Exception as e:
                print(f"Direct API Error: {e}")
                return {"error": "API Fetch Failed"}

            if data and data.get('status') == 'success' and data.get('data', {}).get('candles'):
                candles = data['data']['candles']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            else:
                 return {"error": "No data from Upstox API"}

            # Standardize Columns
            cols = ['open', 'high', 'low', 'close', 'volume']
            # Ensure numeric
            for col in cols:
                df[col] = pd.to_numeric(df[col])

            # Resample if needed
            if resample_rule:
                conversion = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                # Resample and Drop NaN (incomplete intervals?)
                df = df.resample(resample_rule).agg(conversion).dropna()
            
            # --- INDICATOR CALCULATIONS ---
            # Call the shared method
            # Call the shared method
            indicators, indicators_series, pivot_data = await self.calculate_indicators(df, instrument_key=instrument_key, limit=100)
             
            # Prepare final structure
            # Get latest values from the last row
            if len(df) == 0:
                 return {"error": "Insufficient data after resampling"}

            # Define last_row and prev_row for use later
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2] if len(df) > 1 else last_row
            
            # Helper to recursively clean NaNs for JSON safety
            def clean_for_json(obj):
                import math
                if isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                    return obj
                elif isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(i) for i in obj]
                return obj
                  
            # Tail 100 rows for display candles matches limit
            # Format timestamp to ISO string for Frontend consistency
            # Reset index to access timestamp column
            df_reset = df.reset_index()
            df_reset['timestamp'] = df_reset['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S+05:30')
            
            subset = df_reset.tail(100)
            
            # Candles
            display_candles = subset[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
            
            # --- REAL-TIME OVERLAY START ---
            try:
                # Fetch live quote to update LTP and Change (Historical data lags by 1 min)
                live_quotes = self.get_market_quotes([instrument_key])
                live_quote = live_quotes.get(instrument_key)
                
                if live_quote:
                    print(f"Injecting Live Quote for {instrument_key}: {live_quote['ltp']}")
                    # Update LTP in return object
                    last_row_close = live_quote['ltp']
                    live_change = live_quote.get('change', 0)
                    
                    # Create a pseudo-candle for the live tick
                    now_ts = datetime.datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S+05:30')
                    
                    live_candle = {
                        'timestamp': now_ts,
                        'open': live_quote['ltp'], # approximate, strictly should be open of current minute
                        'high': live_quote['ltp'],
                        'low': live_quote['ltp'],
                        'close': live_quote['ltp'],
                        'volume': 0
                    }
                    
                    display_candles.append(live_candle)
                    
                    # Update the return values
                    result = {
                        "ltp": last_row_close,
                        "change": live_change,
                        "candles": display_candles, # Now includes live tick
                        "indicators_series": indicators_series,
                        "indicators": indicators
                    }
                    return clean_for_json(result)
            except Exception as e:
                print(f"Live Quote Injection Failed: {e}")
            # --- REAL-TIME OVERLAY END ---

            result = {
                "ltp": last_row['close'],
                "change": round(last_row['close'] - prev_row['close'], 2),
                "candles": display_candles,
                "indicators_series": indicators_series,
                "indicators": indicators
            }
            cleaned_result = clean_for_json(result)
            print(f"DEBUG: Final Response Keys: {cleaned_result.keys() if cleaned_result else 'None'}")
            print(f"DEBUG: Sample Indicator (RSI): {cleaned_result.get('indicators', {}).get('rsi')}")
            print(f"DEBUG: Pivot Data in Response: {cleaned_result.get('indicators', {}).get('pivot_points')}")
            return cleaned_result

            return {"error": f"Upstox Data Invalid: {data}"}
        except Exception as e:
            print(f"Error fetching Nifty history: {e}")
            import traceback
            return {"error": f"Exception: {str(e)}", "trace": traceback.format_exc()}

    def get_holdings(self):
        try:
            api_client = ApiClient(self.configuration)
            from upstox_client.api.portfolio_api import PortfolioApi
            portfolio_api = PortfolioApi(api_client)
            response = portfolio_api.get_holdings("2.0")
            # Convert list of objects to list of dicts
            data = [h.to_dict() for h in response.data] if response and response.data else []
            return {'data': self._clean_keys(data)}
        except Exception as e:
            print(f"Error fetching holdings: {e}")
            return None

    def get_login_url(self) -> str:
        """
        Generates the login URL for the user to authenticate.
        """
        base_url = "https://api.upstox.com/v2/login/authorization/dialog"
        return f"{base_url}?response_type=code&client_id={self.client_id}&redirect_uri={self.redirect_uri}"

    async def exchange_code_for_token(self, code: str) -> str:
        """
        Exchanges the authorization code for an access token.
        """
        url = "https://api.upstox.com/v2/login/authorization/token"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, data=data)
            response.raise_for_status()
            json_resp = response.json()
            self.access_token = json_resp.get("access_token")
            self.configuration.access_token = self.access_token
            return self.access_token

    async def load_token(self, db):
        """
        Loads the access token from MongoDB on startup.
        """
        try:
            settings_col = db["settings"]
            doc = await settings_col.find_one({"_id": "upstox_config"})
            if doc and "access_token" in doc:
                token = doc["access_token"]
                self.access_token = token
                self.configuration.access_token = token
                print(f"Token loaded from DB: {token[:5]}...")
        except Exception as e:
            print(f"Error loading token from DB: {e}")

    async def save_token(self, db, token: str):
        """
        Saves the access token to MongoDB and updates memory.
        """
        self.access_token = token
        self.configuration.access_token = token
        try:
            settings_col = db["settings"]
            await settings_col.update_one(
                {"_id": "upstox_config"},
                {"$set": {"access_token": token, "updated_at": "now"}},
                upsert=True
            )
            print("Token saved to DB.")
        except Exception as e:
            print(f"Error saving token to DB: {e}")

    def set_access_token(self, token: str):
        """
        Manually sets the access token (Memory only - deprecated for persistent usage).
        """
        self.access_token = token
        self.configuration.access_token = token
        print(f"Access Token set manually: {token[:5]}...")

    def get_market_history(self, instrument_key: str, interval: str, from_date: str, to_date: str):
        """
        Fetches historical candle data.
        """
        if not self.access_token:
            print("Warning: No access token set. Using mock data for testing if allowed, or failing.")
            # For now, let's fail if no token, OR return mock if we want to test without creds.
            # return None
            
        try:
            api_client = ApiClient(self.configuration)
            history_api = HistoryApi(api_client)
            return history_api.get_historical_candle_data1(instrument_key, interval, to_date, from_date, "2.0")
        except Exception as e:
            print(f"Error fetching Upstox data: {e}")
            return None

    def place_order(self, instrument_key: str, quantity: int, transaction_type: str, order_type: str = "MARKET", price: float = 0.0):
        """
        Places an order.
        transaction_type: BUY or SELL
        order_type: MARKET or LIMIT
        """
        if not self.access_token:
            return {"status": "error", "message": "No Access Token"}
            
        try:
            api_client = ApiClient(self.configuration)
            order_api = OrderApi(api_client)
            
            req = PlaceOrderRequest(
                quantity=quantity,
                product="D", # Delivery (D) or Intraday (I) - Defaulting to Delivery for safety/options
                validity="DAY",
                price=price,
                tag="robotrader",
                instrument_token=instrument_key,
                order_type=order_type,
                transaction_type=transaction_type,
                disclosed_quantity=0,
                trigger_price=0.0,
                is_amo=False
            )
            
            response = order_api.place_order(req, "2.0")
            print(f"Order Placed: {response}")
            return {"status": "success", "data": response.data.to_dict() if hasattr(response.data, 'to_dict') else response.data}
        except Exception as e:
            print(f"Error placing order: {e}")
            return {"status": "error", "message": str(e)}

    async def cancel_all_orders(self):
        """
        Cancels all open orders.
        """
        if not self.access_token:
            return {"status": "error", "message": "No Access Token"}
            
        try:
            api_client = ApiClient(self.configuration)
            order_api = OrderApi(api_client)
            
            # 1. Fetch Open Orders
            response = order_api.get_order_book("2.0")
            all_orders = response.data if response and response.data else []
            
            # Debug details
            # print(f"All Orders Statuses: {[o.status for o in all_orders]}")

            # Robust Cancel: Cancel everything NOT in a final state
            final_states = ['complete', 'cancelled', 'rejected', 'expired']
            
            open_orders = [o for o in all_orders if o.status not in final_states]
            
            if not open_orders:
                return {"status": "success", "message": "No open orders to cancel", "count": 0}
                
            print(f"Cancelling {len(open_orders)} orders...")
            cancelled_count = 0
            
            # 2. Cancel Each
            for order in open_orders:
                try:
                    order_api.cancel_order(order.order_id, "2.0")
                    cancelled_count += 1
                except Exception as e:
                    print(f"Failed to cancel {order.order_id}: {e}")
                    
            return {"status": "success", "message": f"Cancelled {cancelled_count} orders", "count": cancelled_count}
            
        except Exception as e:
            print(f"Error cancelling all orders: {e}")
            return {"status": "error", "message": str(e)}

    async def square_off_all_positions(self):
        """
        Squares off all open positions.
        Exit Limit Order = LTP - 0.5 (for Sell) or LTP + 0.5 (for Buy).
        """
        if not self.access_token:
            return {"status": "error", "message": "No Access Token"}
            
        try:
            api_client = ApiClient(self.configuration)
            from upstox_client.api.portfolio_api import PortfolioApi
            p_api = PortfolioApi(api_client)
            
            # 1. Fetch Positions
            response = p_api.get_positions("2.0")
            positions = response.data if response and response.data else []
            
            print(f"Squaring off {len(positions)} raw positions...") # Debug
            
            open_positions = []
            for p in positions:
                # Debug: Print attributes
                # print(f"Pos Object: {p}") 
                # print(f"Dir: {dir(p)}")
                
                # Try getting quantity safely
                qty = getattr(p, 'net_quantity', None)
                if qty is None:
                    # Fallback or check quantity
                    qty = getattr(p, 'quantity', 0)
                
                if qty != 0:
                    open_positions.append(p)
            
            if not open_positions:
                return {"status": "success", "message": "No open positions to square off", "count": 0}
                
            print(f"Squaring off {len(open_positions)} positions (Filtered)...")
            orders_placed = 0
            results = []
            
            # 2. Square Off Each
            for p in open_positions:
                try:
                    # Re-fetch qty safely
                    qty = getattr(p, 'net_quantity', getattr(p, 'quantity', 0))
                    
                    if qty == 0: continue # Should not happen given filter
                    
                    row_net_qty = qty # Store signed quantity for direction check
                    qty = abs(qty) # Absolute for order placement
                    
                    txn_type = "SELL" if row_net_qty > 0 else "BUY"
                    
                    # Logic: User wants LTP - 0.5 (aggressive exit)
                    # We need LTP. Position object usually has 'last_price' or 'ltp' but better to be safe.
                    # Assuming p.last_price is reasonably fresh or fetch fresh?
                    # Upstox positions usually have ltp.
                    ltp = p.last_price
                    
                    price = 0.0
                    if txn_type == "SELL":
                        price = ltp - 0.5
                    else:
                        price = ltp + 0.5 # Symmetric logic for Short Cover
                        
                    # Round price to tick size (0.05)
                    price = round(price * 20) / 20
                    
                    res = self.place_order(
                        instrument_key=p.instrument_token,
                        quantity=qty,
                        transaction_type=txn_type,
                        order_type="LIMIT",
                        price=price
                    )
                    results.append({"key": p.trading_symbol, "status": res.get("status"), "price": price})
                    orders_placed += 1
                except Exception as e:
                    print(f"Failed to square off {p.trading_symbol}: {e}")
                    results.append({"key": p.trading_symbol, "status": "error", "error": str(e)})
                    
            return {"status": "success", "message": f"Placed {orders_placed} exit orders", "results": results}
            
        except Exception as e:
            print(f"Error squaring off: {e}")
            return {"status": "error", "message": str(e)}

    async def square_off_position(self, instrument_key):
        """
        Squares off a specific position by Instrument Key.
        Exit at LTP +/- 0.5.
        """
        if not self.access_token:
            return {"status": "error", "message": "No Access Token"}
            
        try:
            # 1. Fetch ALL positions to find the target (safest way to get Net Qty)
            api_client = ApiClient(self.configuration)
            from upstox_client.api.portfolio_api import PortfolioApi
            p_api = PortfolioApi(api_client)
            
            response = p_api.get_positions("2.0")
            positions = response.data if response and response.data else []
            
            target_pos = None
            for p in positions:
                if p.instrument_token == instrument_key or p.trading_symbol == instrument_key: # Handle both just in case
                    target_pos = p
                    break
            
            if not target_pos:
                return {"status": "error", "message": "Position not found"}
                
            # 2. Calculate Exit Logic
            # Check Quantity (net_quantity or quantity)
            qty = getattr(target_pos, 'net_quantity', getattr(target_pos, 'quantity', 0))
            if qty == 0:
                 return {"status": "error", "message": "Position is already closed (Qty 0)"}

            row_net_qty = qty
            abs_qty = abs(qty)
            txn_type = "SELL" if row_net_qty > 0 else "BUY"
            
            ltp = target_pos.last_price
            if not ltp or ltp == 0:
                 return {"status": "error", "message": "Failed to fetch valid LTP for exit"}
                 
            # User Rule: LTP - 0.5 for SELL, LTP + 0.5 for BUY
            price = ltp - 0.5 if txn_type == "SELL" else ltp + 0.5
            price = round(price * 20) / 20
            
            print(f"Exiting Position: {target_pos.trading_symbol}, Qty: {abs_qty}, Txn: {txn_type}, Price: {price}")
            
            res = self.place_order(
                instrument_key=target_pos.instrument_token,
                quantity=abs_qty,
                transaction_type=txn_type,
                order_type="LIMIT",
                price=price
            )
            return {"status": "success", "message": f"Exit Order Placed for {target_pos.trading_symbol} at {price}", "data": res}
            
        except Exception as e:
            print(f"Error exiting position {instrument_key}: {e}")
            return {"status": "error", "message": str(e)}

    def get_positions(self):
        """
        Fetches current positions (Day + Net).
        """
        if not self.access_token:
            return {"status": "error", "message": "No Access Token"}
            
        try:
            api_client = ApiClient(self.configuration)
            from upstox_client.api.portfolio_api import PortfolioApi
            p_api = PortfolioApi(api_client)
            
            response = p_api.get_positions("2.0")
            print(f"Positions Raw Response: {response}")
            if response and response.data:
                print(f"First Pos Type: {type(response.data[0])}")
                print(f"First Pos Dir: {dir(response.data[0])}")
                
                # Convert SDK objects to dict to ensure serialization
                res_data = [p.to_dict() if hasattr(p, 'to_dict') else vars(p) if hasattr(p, '__dict__') else p for p in response.data]
                print(f"Serialized Data: {res_data}")
                return res_data
            return []
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    def get_market_quotes(self, instrument_keys: list[str]):
        """
        Fetches full market quotes (LTP, OHLC) for a list of instruments.
        Returns a dictionary mapping instrument_key to its quote data.
        """
        if not self.access_token or not instrument_keys:
            return {}

        try:
            # Join keys by comma
            keys_str = ",".join(instrument_keys)
            api_instance = MarketQuoteApi(ApiClient(self.configuration))
            # get_full_market_quote expects instrument_key as comma separated string
            response = api_instance.get_full_market_quote(keys_str, api_version='v2')
            
            if response.status == 'success' and response.data:
                # DEBUG MISMATCH
                req_set = set(instrument_keys)
                res_set = set(response.data.keys())
                print(f"DEBUG: Requested: {req_set}")
                print(f"DEBUG: Response:   {res_set}")
                print(f"DEBUG: Missing:    {req_set - res_set}")
                
                results = {}
                # Debug: Print keys of first item to see structure
                if response.data.values():
                    first_val = list(response.data.values())[0]
                    # print(f"DEBUG: Quote keys: {dir(first_val)}") 
                    # print(f"DEBUG: Quote content: {first_val}")

                for key, quote in response.data.items():
                    # DEBUG: See what we have
                    # print(f"DEBUG: Quote Fields: {dir(quote)}") 
                    # print(f"DEBUG: ISIN: {getattr(quote, 'isin', 'N/A')}")
                    
                    # quote is an object with 'ohlc', 'last_price', 'net_change' etc.
                    # We need to extract relevant fields safely
                    # Structure depends on API v2 response model
                    
                    # Safe extraction helper
                    def get_val(obj, attr, default=0.0):
                        return getattr(obj, attr, default)

                    ohlc = getattr(quote, 'ohlc', None)
                    close = get_val(ohlc, 'close', 0.0) if ohlc else 0.0
                    open_price = get_val(ohlc, 'open', 0.0) if ohlc else 0.0
                    high = get_val(ohlc, 'high', 0.0) if ohlc else 0.0
                    low = get_val(ohlc, 'low', 0.0) if ohlc else 0.0
                    
                    ltp = get_val(quote, 'last_price', 0.0)
                    
                    # Try getting net_change directly if available
                    net_change = get_val(quote, 'net_change', None)
                    
                    change = 0.0
                    change_percent = 0.0
                    
                    if net_change is not None:
                        change = float(net_change)
                    elif close > 0:
                        change = ltp - close
                    
                    if close > 0:
                        change_percent = (change / close) * 100
                    
                    quote_data = {
                        "ltp": ltp,
                        "close": close,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2)
                    }
                    
                    # 1. Primary Key Normalization (Colon -> Pipe)
                    final_key = key.replace(':', '|')
                    results[final_key] = quote_data
                    
                    # 2. Raw Key (Just in case)
                    if key != final_key:
                        results[key] = quote_data

                    # 3. Aliasing from Quote Details (ISIN, Token)
                    # We try to index this quote by every possible identifier we can find
                    # so that whatever key the user used, we match it.
                    
                    # ISIN
                    isin = getattr(quote, 'isin', None)
                    if isin:
                        # Assuming NSE_EQ for now, or match prefix from key
                        # Logic: If Request was "NSE_EQ|ISIN", we want "NSE_EQ|ISIN" to map to this.
                        # We can construct keys: "NSE_EQ|{isin}", "BSE_EQ|{isin}"
                        # or just rely on prefix from 'key' if available.
                        prefix = final_key.split('|')[0] if '|' in final_key else "NSE_EQ"
                        results[f"{prefix}|{isin}"] = quote_data
                        results[f"{prefix}:{isin}"] = quote_data
                        
                    # Instrument Token
                    token = getattr(quote, 'instrument_token', None)
                    if token:
                         prefix = final_key.split('|')[0] if '|' in final_key else "NSE_EQ"
                         results[f"{prefix}|{token}"] = quote_data
                         results[f"{prefix}:{token}"] = quote_data
                         # Also Index by raw token just in case fallback checks token only
                         results[str(token)] = quote_data

                return results
            return {}
        except Exception as e:
            error_msg = f"Error fetching market quotes: {e}"
            print(error_msg)
            return {"_debug_error": str(e)}

upstox_service = UpstoxService()
