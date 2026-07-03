import logging
import sys
import traceback
from tradingapi_a.mconnect import MConnect
from datetime import datetime, timedelta

class MiraeService:
    # Maps the exchange field returned by broker position API -> exchange used in order placement
    # e.g. NSE equity options are placed on NFO, BSE equity options on BFO
    EXCHANGE_MAP = {
        "NSE": "NFO",
        "BSE": "BFO",
        "NFO": "NFO",
        "BFO": "BFO",
        "MCX": "MCX",
        "CDS": "CDS",
    }

    # Valid mStock variety (goes in URL path): regular, amo, co
    # Valid mStock order types: MARKET, LIMIT, SL, SL-M
    # Valid product: CNC, NRML, MIS, MTF
    # Valid validity: DAY, IOC

    def __init__(self):
        self.mconnect = None
        self.logger = logging.getLogger(__name__)
        self.token_symbol_map = {}
        self.prev_close_cache = {}


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
            import requests as _requests
            url = self.mconnect.default_root_uri.rstrip("/") + "/" + self.mconnect.routes["net_position"].lstrip("/")
            headers = {
                "X-Mirae-Version": "1",
                "Authorization": f"token {self.mconnect.api_key}:{self.mconnect.access_token}",
                "Content-Type": "application/json"
            }
            res = _requests.get(url, headers=headers, timeout=15)
            self.logger.info(f"get_net_position: status={res.status_code}, body_len={len(res.content)}")

            if res.status_code == 200:
                if not res.content or res.content.strip() == b'':
                    # HTTP 200 with empty body = flat book (no positions)
                    self.logger.info("get_net_position: 200 with empty body — no open positions")
                    return True, {"status": "success", "data": {"net": [], "day": None}}
                try:
                    return True, res.json()
                except Exception as e:
                    self.logger.error(f"get_net_position: could not parse JSON: {e}, body={res.text[:200]}")
                    return False, f"Could not parse positions response: {e}"
            else:
                self.logger.error(f"get_net_position: HTTP {res.status_code}, body={res.text[:300]}")
                return False, f"Positions API returned HTTP {res.status_code}: {res.text[:200]}"

        except Exception as e:
            self.logger.error(f"get_net_position exception: {e}")
            return False, str(e)


    def cancel_all_orders(self):
        if not self.mconnect:
            return False, "Mirae service not initialized"
        try:
            res = self.mconnect.cancel_all()
            if hasattr(res, 'content'):
                if not res.content or res.content.strip() == b'':
                    return True, {"status": "success", "message": "No orders to cancel"}
                try:
                    return True, res.json()
                except Exception:
                    return True, {"status": "success", "message": "No orders to cancel"}
            return True, res
        except Exception as e:
            err = str(e)
            if "parse" in err.lower() or "b''" in err or "empty" in err.lower():
                return True, {"status": "success", "message": "No orders to cancel"}
            return False, err

    def get_funds(self):
        if not self.mconnect:
            return False, "Mirae service not initialized"
        try:
            res = self.mconnect.get_fund_summary()
            if hasattr(res, 'content'):
                if not res.content or res.content.strip() == b'':
                    return True, {"status": "success", "data": {}}
                try:
                    return True, res.json()
                except Exception:
                    return True, {"status": "success", "data": {}}
            return True, res
        except Exception as e:
            err = str(e)
            if "parse" in err.lower() or "b''" in err or "empty" in err.lower():
                return True, {"status": "success", "data": {}}
            return False, e

    def _adjust_year(self, dt):
        if dt.year < 2026:
            try:
                return dt.replace(year=dt.year + 10)
            except ValueError:
                # Handle leap year day (Feb 29)
                return dt.replace(year=dt.year + 10, day=28)
        return dt

    def get_lot_size(self, index_name: str) -> int:
        idx = index_name.upper()
        if "NIFTY 50" in idx or ("NIFTY" in idx and "BANK" not in idx and "FIN" not in idx and "MID" not in idx): return 65
        if "BANKNIFTY" in idx or "BANK" in idx: return 30
        if "FINNIFTY" in idx or "FIN" in idx: return 65
        if "MID" in idx or "MIDCPNIFTY" in idx: return 120
        if "SENSEX" in idx: return 20
        if "BANKEX" in idx: return 30
        return 50

    def get_expiry_dates(self, index_name: str) -> list:
        if not self.mconnect:
            return []
        try:
            exchange_id = "5" if ("sensex" in index_name.lower() or "bankex" in index_name.lower()) else "2"
            res = self.mconnect.get_option_chain_master(exchange_id)
            if res.status_code == 200:
                data = res.json()
                nested_data = data.get("data", {})
                dct_exp = nested_data.get("dctExp", {})
                
                expiries = []
                for ts in dct_exp.values():
                    dt = datetime.fromtimestamp(int(ts))
                    dt = self._adjust_year(dt)
                    expiries.append(dt.strftime("%Y-%m-%d"))
                return sorted(list(set(expiries)))
            return []
        except Exception as e:
            self.logger.error(f"Error fetching Mirae expiries: {e}")
            return []

    def get_option_chain(self, index_name: str, expiry_date: str) -> dict:
        if not self.mconnect:
            return {}
        try:
            exchange_id = "5" if ("sensex" in index_name.lower() or "bankex" in index_name.lower()) else "2"
            
            token = "26000" # Nifty 50 default
            if "banknifty" in index_name.lower() or "nifty bank" in index_name.lower():
                token = "26009"
            elif "finnifty" in index_name.lower() or "fin" in index_name.lower():
                token = "26037"
            elif "midcpnifty" in index_name.lower() or "mid" in index_name.lower():
                token = "26074"
            elif "sensex" in index_name.lower():
                token = "51"
            elif "bankex" in index_name.lower():
                token = "69"
            
            import time
            from datetime import datetime
            
            # Query expiries from master to find matching timestamp
            master_res = self.mconnect.get_option_chain_master(exchange_id)
            expiry_ts = None
            if master_res.status_code == 200:
                dct_exp = master_res.json().get("data", {}).get("dctExp", {})
                for ts in dct_exp.values():
                    dt = datetime.fromtimestamp(int(ts))
                    dt = self._adjust_year(dt)
                    if dt.strftime("%Y-%m-%d") == expiry_date:
                        expiry_ts = str(ts)
                        break
            
            # Fallback to local conversion if master query failed
            if not expiry_ts:
                dt = datetime.strptime(expiry_date, "%Y-%m-%d")
                expiry_ts = str(int(time.mktime(dt.timetuple())))

            res = self.mconnect.get_option_chain_data(exchange_id, expiry_ts, token)
            if res.status_code == 200:
                raw_data = res.json()
                nested_data = raw_data.get("data", {})
                
                contract_model = nested_data.get("contractModel", {})
                sym = contract_model.get("sym", "NIFTY")
                exp = contract_model.get("exp")
                
                # Parse date to check if it's the last weekday of the month
                is_monthly = False
                dt = None
                if exp:
                    try:
                        dt = datetime.strptime(exp, "%d-%b-%Y")
                        # Add timedelta
                        from datetime import timedelta
                        next_week = dt + timedelta(days=7)
                        is_monthly = (next_week.month != dt.month)
                    except Exception as ex:
                        self.logger.error(f"Error parsing expiry date: {ex}")
                
                chain_list = []
                total_ce = 0
                total_pe = 0
                
                # Helper to format symbol to mStock format
                def format_symbol(strike_val, opt_type):
                    if not dt:
                        return f"{sym}-{strike_val}-{opt_type}"
                    yy = dt.strftime("%y")
                    if is_monthly:
                        mmm = dt.strftime("%b").upper() # "JUL"
                        return f"{sym}{yy}{mmm}{int(strike_val)}{opt_type}"
                    else:
                        m_code = str(dt.month)
                        if dt.month == 10: m_code = "O"
                        elif dt.month == 11: m_code = "N"
                        elif dt.month == 12: m_code = "D"
                        dd = dt.strftime("%d")
                        return f"{sym}{yy}{m_code}{dd}{int(strike_val)}{opt_type}"
                
                opt_exchange = "BFO" if exchange_id == "5" else "NFO"

                # Parse Call Options
                calls = nested_data.get("call", [])
                for item in calls:
                    parts = item.split(",")
                    if len(parts) >= 3:
                        c_token = parts[0]
                        strike_price = float(parts[1]) / 100
                        oi = int(parts[2])
                        total_ce += oi
                        
                        trading_symbol = format_symbol(strike_price, "CE")
                        self.token_symbol_map[c_token] = trading_symbol
                        
                        chain_list.append({
                            'strike_price': float(strike_price),
                            'instrument_type': 'CE',
                            'instrument_key': f"{opt_exchange}:{c_token}",
                            'trading_symbol': trading_symbol,
                            'name': trading_symbol,
                            'last_price': 0.0,
                            'open_interest': int(oi),
                            'volume': 0,
                            'price_change': 0.0,
                            'bid_price': 0.0,
                            'ask_price': 0.0,
                            'low_price': 0.0,
                            'iv': 0.0,
                            'delta': 0.0,
                            'oi_value': 0.0,
                            'lot_size': self.get_lot_size(index_name)
                        })
                        
                # Parse Put Options
                puts = nested_data.get("put", [])
                for item in puts:
                    parts = item.split(",")
                    if len(parts) >= 3:
                        p_token = parts[0]
                        strike_price = float(parts[1]) / 100
                        oi = int(parts[2])
                        total_pe += oi
                        
                        trading_symbol = format_symbol(strike_price, "PE")
                        self.token_symbol_map[p_token] = trading_symbol
                        
                        chain_list.append({
                            'strike_price': float(strike_price),
                            'instrument_type': 'PE',
                            'instrument_key': f"{opt_exchange}:{p_token}",
                            'trading_symbol': trading_symbol,
                            'name': trading_symbol,
                            'last_price': 0.0,
                            'open_interest': int(oi),
                            'volume': 0,
                            'price_change': 0.0,
                            'bid_price': 0.0,
                            'ask_price': 0.0,
                            'low_price': 0.0,
                            'iv': 0.0,
                            'delta': 0.0,
                            'oi_value': 0.0,
                            'lot_size': self.get_lot_size(index_name)
                        })

                # Fetch real-time LTPs in batch for all options + spot index
                ltp_queries = []
                opt_exchange = "BFO" if exchange_id == "5" else "NFO"
                query_to_item = {}
                
                for item in chain_list:
                    symbol = item['trading_symbol']
                    query_str = f"{opt_exchange}:{symbol}"
                    ltp_queries.append(query_str)
                    query_to_item[query_str] = item
                
                spot_symbol = "NSE:Nifty 50"
                if "banknifty" in index_name.lower() or "nifty bank" in index_name.lower():
                    spot_symbol = "NSE:Nifty Bank"
                elif "finnifty" in index_name.lower() or "fin" in index_name.lower():
                    spot_symbol = "NSE:Nifty Fin Service"
                elif "mid" in index_name.lower():
                    spot_symbol = "NSE:Nifty Mid Select"
                elif "sensex" in index_name.lower():
                    spot_symbol = "BSE:SENSEX"
                elif "bankex" in index_name.lower():
                    spot_symbol = "BSE:BANKEX"
                
                ltp_queries.append(spot_symbol)

                # Fetch missing close prices once for cache
                if len(self.prev_close_cache) > 2000:
                    self.prev_close_cache.clear()
                
                missing_close = [q for q in ltp_queries if q not in self.prev_close_cache]
                if missing_close:
                    chunk_size = 50
                    for i in range(0, len(missing_close), chunk_size):
                        chunk = missing_close[i:i + chunk_size]
                        try:
                            ohlc_res = self.mconnect.get_ohlc(chunk)
                            if ohlc_res.status_code == 200:
                                ohlc_data = ohlc_res.json().get("data", {})
                                if isinstance(ohlc_data, dict):
                                    for sym_key, sym_data in ohlc_data.items():
                                        close_val = sym_data.get("close")
                                        if close_val is None:
                                            # Fallback check for nested ohlc dict
                                            close_val = sym_data.get("ohlc", {}).get("close")
                                        if close_val is not None:
                                            self.prev_close_cache[sym_key] = float(close_val)
                        except Exception as e:
                            self.logger.error(f"Error fetching OHLC close cache: {e}")
                
                # Batch query LTPs (fast poll)
                ltp_map = {}
                chunk_size = 50
                for i in range(0, len(ltp_queries), chunk_size):
                    chunk = ltp_queries[i:i + chunk_size]
                    try:
                        ltp_res = self.mconnect.get_ltp(chunk)
                        if ltp_res.status_code == 200:
                            res_data = ltp_res.json().get("data", {})
                            if isinstance(res_data, dict):
                                ltp_map.update(res_data)
                    except Exception as e:
                        self.logger.error(f"Error fetching LTP chunk: {e}")
                
                # Update prices and calculate daily change
                for query_str, price_data in ltp_map.items():
                    if query_str in query_to_item:
                        item = query_to_item[query_str]
                        ltp = float(price_data.get("last_price", 0.0))
                        item['last_price'] = ltp
                        item['bid_price'] = round(ltp - 0.05, 2) if ltp > 0 else 0.0
                        item['ask_price'] = round(ltp + 0.05, 2) if ltp > 0 else 0.0
                        item['oi_value'] = item['open_interest'] * ltp

                        # Calculate price change compared to previous close
                        close = self.prev_close_cache.get(query_str, 0.0)
                        if close > 0:
                            price_change = round(ltp - close, 2)
                            pchange = round((price_change / close) * 100, 2)
                        else:
                            price_change = 0.0
                            pchange = 0.0
                        item['price_change'] = price_change
                        item['pchange'] = pchange


                spot_price = 0.0
                if spot_symbol in ltp_map:
                    spot_price = float(ltp_map[spot_symbol].get("last_price", 0.0))
                
                if spot_price == 0.0 and chain_list:
                    strikes = sorted(list(set([x['strike_price'] for x in chain_list])))
                    if strikes:
                        spot_price = strikes[len(strikes) // 2]
                
                # Calculate spot index daily change metrics
                spot_change = 0.0
                spot_pchange = 0.0
                spot_close = self.prev_close_cache.get(spot_symbol, 0.0)
                if spot_close > 0 and spot_price > 0:
                    spot_change = round(spot_price - spot_close, 2)
                    spot_pchange = round((spot_change / spot_close) * 100, 2)

                return {
                    "chain": chain_list,
                    "spot_price": spot_price,
                    "spot_change": spot_change,
                    "spot_pchange": spot_pchange,
                    "totals": {"ce": total_ce, "pe": total_pe}
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching Mirae option chain: {e}")
            return {}

    def _translate_position_symbol(self, trading_symbol, src_exchange):
        """Translate the hyphenated position symbol (e.g. SENSEX-02Jul2026-77400-CE) to mStock order format.
        Returns (order_symbol, order_exchange)."""
        # Resolve numeric token IDs from cache first
        if str(trading_symbol).isdigit():
            resolved = self.token_symbol_map.get(str(trading_symbol))
            if resolved:
                self.logger.info(f"Resolved token ID {trading_symbol} -> {resolved}")
                trading_symbol = resolved

        # Auto-detect correct exchange from trading symbol prefix to prevent mismatch
        ts_upper = str(trading_symbol).upper()
        if "SENSEX" in ts_upper or "BANKEX" in ts_upper:
            src_exchange = "BFO"
        elif "NIFTY" in ts_upper or "FINNIFTY" in ts_upper or "MIDCPNIFTY" in ts_upper:
            src_exchange = "NFO"

        # Dynamically resolve the order exchange from position exchange using the lookup table
        order_exchange = self.EXCHANGE_MAP.get(src_exchange, src_exchange)

        # Parse hyphenated format: INDEX-DDMonYYYY-STRIKE-OPTTYPE
        parts = trading_symbol.split("-")
        if len(parts) == 4:
            index_sym = parts[0]
            expiry_str = parts[1]  # e.g. 02Jul2026
            strike = parts[2]
            opt_type = parts[3]    # CE or PE
            try:
                dt = datetime.strptime(expiry_str, "%d%b%Y")
                yy = dt.strftime("%y")
                next_week = dt + timedelta(days=7)
                is_monthly = (next_week.month != dt.month)
                if is_monthly:
                    mmm = dt.strftime("%b").upper()
                    order_symbol = f"{index_sym}{yy}{mmm}{int(float(strike))}{opt_type}"
                else:
                    m_code = str(dt.month)
                    if dt.month == 10: m_code = "O"
                    elif dt.month == 11: m_code = "N"
                    elif dt.month == 12: m_code = "D"
                    dd = dt.strftime("%d")
                    order_symbol = f"{index_sym}{yy}{m_code}{dd}{int(float(strike))}{opt_type}"
                self.logger.info(f"Translated '{trading_symbol}' -> '{order_symbol}' on {order_exchange}")
                return order_symbol, order_exchange
            except Exception as e:
                self.logger.warning(f"Could not translate symbol '{trading_symbol}': {e}")

        return trading_symbol, order_exchange

    def place_order(self, variety, tradingsymbol, exchange, transaction_type, order_type, quantity, product, validity, price, trigger_price, disclosed_quantity=0, tag="RoboTrader"):
        if not self.mconnect:
            return {"status": "error", "message": "Mirae service not initialized"}
        try:
            import requests as _requests

            # Resolve numeric token IDs to trading symbols using the cached option chain map
            if str(tradingsymbol).isdigit() and str(tradingsymbol) in self.token_symbol_map:
                resolved_symbol = self.token_symbol_map[str(tradingsymbol)]
                self.logger.info(f"Resolved token ID {tradingsymbol} to trading symbol {resolved_symbol}")
                tradingsymbol = resolved_symbol

            # Normalize order type aliases
            normalized_order_type = order_type
            if order_type in ("MKT", "MKT_ORDER"):
                normalized_order_type = "MARKET"
            elif order_type in ("LMT",):
                normalized_order_type = "LIMIT"

            # variety goes in the URL path: POST /openapi/typea/orders/{variety}
            # Valid values: NORMAL, AMO, COVER
            base = self.mconnect.default_root_uri.rstrip("/")
            url = f"{base}/openapi/typea/orders/{variety}"

            headers = {
                "X-Mirae-Version": "1",
                "Authorization": f"token {self.mconnect.api_key}:{self.mconnect.access_token}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            data = {
                "tradingsymbol": tradingsymbol,
                "exchange": exchange,
                "transaction_type": transaction_type,
                "order_type": normalized_order_type,
                "quantity": str(int(quantity)),
                "product": product,
                "validity": validity,
                "price": str(price),
                "trigger_price": str(trigger_price),
                "disclosed_quantity": str(disclosed_quantity),
                "tag": tag,
            }

            self.logger.info(f"place_order RAW: POST {url} | data={data}")
            res = _requests.post(url, headers=headers, data=data, timeout=15)
            self.logger.info(f"place_order response: status={res.status_code} body={res.text[:300]}")

            if res.content:
                return res.json()
            return {"status": "error", "message": f"Empty response from broker (HTTP {res.status_code})"}
        except Exception as e:
            self.logger.error(f"place_order exception: {e}")
            return {"status": "error", "message": str(e)}

    def square_off_position(self, symbol, quantity, transaction_type, product, exchange="NFO"):
        opposite_type = "SELL" if transaction_type == "BUY" else "BUY"
        # Translate hyphenated broker symbol to standard mStock order format
        order_symbol, order_exchange = self._translate_position_symbol(symbol, exchange)
        return self.place_order(
            variety="regular",
            tradingsymbol=order_symbol,
            exchange=order_exchange,
            transaction_type=opposite_type,
            order_type="MARKET",
            quantity=quantity,
            product=product,
            validity="DAY",
            price=0.0,
            trigger_price=0.0
        )

    def square_off_all_positions(self):
        success, positions_data = self.get_net_position()
        if not success:
            return {"status": "error", "message": f"Failed to get positions: {positions_data}"}
        
        results = []
        positions_list = []
        if isinstance(positions_data, list):
            positions_list = positions_data
        elif isinstance(positions_data, dict):
            data_val = positions_data.get("data", positions_data)
            if isinstance(data_val, dict):
                positions_list = data_val.get("net", []) or []
            elif isinstance(data_val, list):
                positions_list = data_val
            
        for pos in positions_list:
            buy_qty = int(pos.get("buy_quantity", 0) or 0)
            sell_qty = int(pos.get("sell_quantity", 0) or 0)
            net_qty = buy_qty - sell_qty
            if net_qty != 0:
                symbol = pos.get("tradingsymbol") or pos.get("tradingSymbol") or pos.get("symbol")
                exchange = pos.get("exchange", "NFO")
                prod = pos.get("product") or pos.get("productType") or "MIS"
                side = "SELL" if net_qty > 0 else "BUY"
                abs_qty = abs(net_qty)
                res = self.square_off_position(symbol, abs_qty, side, prod, exchange)
                results.append({"symbol": symbol, "result": res})
        return {"status": "success", "results": results}

    def get_trade_book(self):
        if not self.mconnect:
            return False, "Mirae service not initialized"
        try:
            res = self.mconnect.get_trade_book()
            if hasattr(res, 'json'):
                return True, res.json()
            return True, res
        except Exception as e:
            return False, str(e)

mirae_service = MiraeService()
