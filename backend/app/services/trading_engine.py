import asyncio
import pandas as pd
from datetime import datetime
from .indicators import indicators
from .upstox_service import upstox_service
from .mock_trade_service import mock_trade_service
from .telegram_service import telegram_service
import pandas_ta as ta
import numpy as np

class TradingEngine:
    def __init__(self):
        self.active = False
        self.strategies = []
        self.mongodb = None # Will be set by main.py
        self.instrument_key = "NSE_INDEX|Nifty 50" # Default instrument
        self.last_run = None
        self.loop_task = None
        self.instrument_key = "NSE_INDEX|Nifty 50" # Default instrument
        self.last_run = None
        self.loop_task = None

    def set_db(self, db):
        self.mongodb = db

    async def start_trading(self):
        if self.active:
            print("Trading already active")
            return
        self.active = True
        print("Trading Engine Started")
        self.loop_task = asyncio.ensure_future(self.run_loop())

    def stop_trading(self):
        self.active = False
        print("Trading Engine Stopped")

    async def run_loop(self):
        print("Starting Trading Engine Loop...")
        while self.active:
            try:
                self.last_run = datetime.now()
                # Process deployed strategies
                await self.process_deployments()
            except Exception as e:
                print(f"CRITICAL ERROR in Trading Engine: {e}")
            
            await asyncio.sleep(10) # Fetch every 10 seconds for live monitoring

    async def process_deployments(self):
        if self.mongodb is None:
            self._debug_log("Engine: MongoDB not set, skipping")
            return

        # Check Trading Hours Constraint
        try:
            general_config = await self.mongodb["settings"].find_one({"id": "general_config"})
            if general_config and general_config.get("enforce_market_hours", False):
                now = datetime.now().time()
                import datetime as dt
                market_start = dt.time(9, 0)
                market_end = dt.time(15, 30)
                if not (market_start <= now <= market_end):
                    # print("Outside market hours. Skipping deployment evaluations.")
                    return
        except Exception as e:
            print(f"Error checking market hours: {e}")

        # 1. Fetch Active Deployments
        try:
            cursor = self.mongodb["strategy_deployments"].find({"status": "ACTIVE"})
            deployments = await cursor.to_list(length=100)
        except Exception as e:
            print(f"Error fetching deployments: {e}")
            return

        # 2. Evaluate deployments concurrently with a concurrency limit
        sem = asyncio.Semaphore(5) # Limit to 5 concurrent evaluations to avoid API rate limits

        async def _safe_evaluate(dep):
            async with sem:
                try:
                    await asyncio.wait_for(self.evaluate_deployment(dep), timeout=25.0)
                except asyncio.TimeoutError:
                    print(f"TIMEOUT evaluating deployment {dep.get('_id')}")
                except Exception as e:
                    print(f"Error evaluating deployment {dep.get('_id')}: {e}")
                    if self.mongodb is not None:
                        try:
                            await self.mongodb["deployment_logs"].insert_one({
                                "deployment_id": str(dep["_id"]),
                                "timestamp": datetime.now(),
                                "instrument": dep.get('primary_instrument', 'unknown'),
                                "interval": dep.get('interval', 'N/A'),
                                "close_price": 0,
                                "signal": False,
                                "traded": False,
                                "rules": [],
                                "error": str(e)
                            })
                        except Exception as le:
                            print(f"Double error! Failed to log error to DB: {le}")

        tasks = [_safe_evaluate(dep) for dep in deployments]
        if tasks:
            await asyncio.gather(*tasks)

    def _to_json_safe(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, list):
            return [self._to_json_safe(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._to_json_safe(v) for k, v in obj.items()}
        return obj

    async def evaluate_deployment(self, dep):
        # Determine instruments to monitor
        is_advanced = dep.get('is_advanced', False)
        execution_plan = dep.get('execution_plan', [])
        saved_strategies = dep.get('saved_strategies', [])
        
        ik = dep.get('primary_instrument')
        interval = dep.get('interval', '15minute')
        if is_advanced and execution_plan:
            ik = execution_plan[0].get('leg')
            interval = execution_plan[0].get('timeframe')

        if not ik:
            return

        # 1. Fetch Latest Data
        try:
            df = await upstox_service._fetch_historical_df(ik, interval, days_back_override=2)
        except Exception as e:
            raise e

        if df is None or df.empty or len(df) < 30:
            candle_count = 0 if df is None else len(df)
            error_msg = f"Data fetch failed for {ik}: got {candle_count} candles (need 30+). Check Upstox token."
            if self.mongodb is not None:
                await self.mongodb["deployment_logs"].insert_one({
                    "deployment_id": str(dep["_id"]),
                    "timestamp": datetime.now(),
                    "instrument": ik,
                    "interval": interval,
                    "close_price": 0,
                    "signal": False, "traded": False,
                    "rules": [],
                    "error": error_msg
                })
            return

        # 2. Calculate Indicators
        try:
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
        except Exception as e:
            raise e

        # 3. Evaluate Logic
        has_signal = False
        rule_results = []
        latest_close = float(df['close'].iloc[-1])
        
        if not is_advanced:
            # Default Strategy Logic (Same as backtester simple mode)
            try:
                bbl_col = [c for c in df.columns if c.startswith('BBL_')][0]
                stoch_k_col = [c for c in df.columns if c.startswith('STOCHk_')][0]
                stoch_d_col = [c for c in df.columns if c.startswith('STOCHd_')][0]
                macd_h_col = [c for c in df.columns if c.startswith('MACDh_')][0]

                cond1 = bool(df['low'].iloc[-1] <= df[bbl_col].iloc[-1])
                cond2 = bool(df[stoch_k_col].iloc[-1] < 20)
                cond3 = bool((df[stoch_k_col].iloc[-1] > df[stoch_d_col].iloc[-1]) and (df[stoch_k_col].iloc[-2] <= df[stoch_d_col].iloc[-2]))
                cond4 = bool(df[macd_h_col].iloc[-1] > df[macd_h_col].iloc[-2])
                
                has_signal = cond1 and cond2 and cond3 and cond4
                rule_results = [
                    {"rule": "Low <= BBL", "matched": cond1, "left": round(float(df['low'].iloc[-1]), 2), "right": round(float(df[bbl_col].iloc[-1]), 2)},
                    {"rule": "StochK < 20", "matched": cond2, "left": round(float(df[stoch_k_col].iloc[-1]), 2), "right": 20},
                    {"rule": "StochK crosses above StochD", "matched": cond3, "left": round(float(df[stoch_k_col].iloc[-1]), 2), "right": round(float(df[stoch_d_col].iloc[-1]), 2)},
                    {"rule": "MACD Histogram rising", "matched": cond4, "left": round(float(df[macd_h_col].iloc[-1]), 4), "right": round(float(df[macd_h_col].iloc[-2]), 4)},
                ]
            except Exception as e:
                raise e
        else:
            strat_id = execution_plan[0].get('strategyId')
            strat = next((s for s in saved_strategies if str(s['id']) == str(strat_id)), None)
            
            if strat and 'rules' in strat:
                match = True
                for rule in strat['rules']:
                    try:
                        left_val = self._get_val(df, rule['indicator'], -1)
                        right_val = float(rule['value']) if rule['valueType'] == 'number' else self._get_val(df, rule['value'], -1)
                        op = rule['operator']
                        rule_match = False
                        
                        if op == '>': rule_match = left_val > right_val
                        elif op == '<': rule_match = left_val < right_val
                        elif op == '>=': rule_match = left_val >= right_val
                        elif op == '<=': rule_match = left_val <= right_val
                        elif op == 'cross_above':
                            l_prev = self._get_val(df, rule['indicator'], -2)
                            r_prev = float(rule['value']) if rule['valueType'] == 'number' else self._get_val(df, rule['value'], -2)
                            rule_match = left_val > right_val and l_prev <= r_prev
                        elif op == 'cross_below':
                            l_prev = self._get_val(df, rule['indicator'], -2)
                            r_prev = float(rule['value']) if rule['valueType'] == 'number' else self._get_val(df, rule['value'], -2)
                            rule_match = left_val < right_val and l_prev >= r_prev
                        elif op == '==': rule_match = left_val == right_val
                        else: rule_match = True
                        
                        match = match and rule_match
                        rule_results.append({
                            "rule": f"{rule['indicator']} {op} {rule['value']}",
                            "matched": bool(rule_match),
                            "left": round(float(left_val), 4) if not isinstance(left_val, str) else left_val,
                            "right": round(float(right_val), 4) if not isinstance(right_val, str) else right_val,
                        })
                    except Exception as e:
                        match = False
                        rule_results.append({"rule": f"{rule.get('indicator','')} {rule.get('operator','')} {rule.get('value','')}", "matched": False, "error": str(e)})
                has_signal = match

        # 4. Log the evaluation to deployment_logs
        traded = False
        if has_signal:
            print(f"SIGNAL DETECTED for {ik}")
            await self.execute_trade(dep, ik, latest_close)
            traded = True
        
        if self.mongodb is not None:
            log_entry = {
                "deployment_id": str(dep["_id"]),
                "timestamp": datetime.now(),
                "instrument": ik,
                "interval": interval,
                "close_price": latest_close,
                "signal": bool(has_signal),
                "traded": traded,
                "rules": self._to_json_safe(rule_results),
            }
            await self.mongodb["deployment_logs"].insert_one(log_entry)

    def _get_val(self, df, name, idx):
        is_prev = name.endswith('_prev')
        base_name = name.replace('_prev', '')
        lookup_idx = idx - 1 if is_prev else idx
        
        col_map = {
            'close': 'close', 'open': 'open', 'low': 'low', 'high': 'high',
            'STOCHk': next((c for c in df.columns if c.startswith('STOCHk_')), 'close'),
            'STOCHd': next((c for c in df.columns if c.startswith('STOCHd_')), 'close'),
            'MACDh': next((c for c in df.columns if c.startswith('MACDh_')), 'close'),
            'BBL_20_2': next((c for c in df.columns if c.startswith('BBL_')), 'close'),
            'STOCHRSIk': next((c for c in df.columns if c.startswith('STOCHRSIk_')), 'close'),
        }
        return df[col_map.get(base_name, 'close')].iloc[lookup_idx]

    async def execute_trade(self, dep, signal_instrument_key, signal_price):
        trade_instrument_key = dep.get('trade_instrument_key') or signal_instrument_key
        mode = dep.get('deployment_mode', 'MOCK')
        side = "BUY" if dep.get('trade_type', 'LONG') == 'LONG' else "SELL"
        
        trade_price = signal_price
        if trade_instrument_key != signal_instrument_key:
            # Fetch latest price for the trade instrument
            try:
                quotes = upstox_service.get_market_quotes([trade_instrument_key])
                if trade_instrument_key in quotes:
                    trade_price = quotes[trade_instrument_key].get('ltp', signal_price)
            except:
                print(f"Engine: Failed to fetch quote for {trade_instrument_key}, using signal price")

        qty = dep.get('quantity', 1)
        quantity_type = dep.get('quantity_type', 'MANUAL')
        capital_to_use = dep.get('capital_to_use', 0.0)
        capital_percentage = dep.get('capital_percentage', 0.0)
        lot_size = dep.get('lot_size', 1)
        
        calculated_capital = capital_to_use

        if quantity_type == "PERCENTAGE" and capital_percentage > 0:
            try:
                funds = upstox_service.get_funds()
                if funds and isinstance(funds, dict) and 'equity' in funds:
                    avail_margin = float(funds['equity'].get('available_margin', 0.0))
                    calculated_capital = avail_margin * (capital_percentage / 100.0)
                    print(f"Engine: Upstox Available Margin: {avail_margin}, using {capital_percentage}% = {calculated_capital}")
                else:
                    print(f"Engine: Could not fetch Upstox funds for PERCENTAGE calculation.")
            except Exception as e:
                print(f"Engine: Error fetching Upstox funds: {e}")

        if quantity_type in ["CAPITAL", "PERCENTAGE"] and calculated_capital > 0 and trade_price > 0 and lot_size > 0:
            import math
            cost_per_lot = lot_size * trade_price
            if cost_per_lot > 0:
                num_lots = math.floor(calculated_capital / cost_per_lot)
                qty = num_lots * lot_size
                if qty <= 0:
                    msg = f"⚠️ <b>Trade Skipped</b>\nAllocated Capital (₹{calculated_capital:.2f}) is too low to buy 1 lot. Cost/Lot: ₹{cost_per_lot:.2f}"
                    print(f"Engine: {msg}")
                    try:
                        await telegram_service.send_message(msg)
                    except: pass
                    return # Skip trade

        print(f"Engine: Executing {side} {qty} {trade_instrument_key} at {trade_price} ({mode})")
        
        if mode == "MOCK":
            await mock_trade_service.place_order({
                "instrument_key": trade_instrument_key,
                "quantity": qty,
                "transaction_type": side,
                "price": trade_price
            })
        else:
            # LIVE Upstox Trade
            await upstox_service.place_order(
                instrument_key=trade_instrument_key,
                quantity=qty,
                transaction_type=side,
                order_type="MARKET"
            )
            
        # Update deployment status to COMPLETED (Single trade per deployment for now)
        await self.mongodb["strategy_deployments"].update_one(
            {"_id": dep["_id"]},
            {"$set": {"status": "TRADED", "last_trade_at": datetime.now()}}
        )

        # Telegram Notification
        msg = (
            f"🚀 <b>Trade Executed!</b>\n\n"
            f"<b>Instrument:</b> {trade_instrument_key}\n"
            f"<b>Side:</b> {side}\n"
            f"<b>Quantity:</b> {qty}\n"
            f"<b>Price:</b> ₹{trade_price}\n"
            f"<b>Mode:</b> {mode}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await telegram_service.send_message(msg)

trading_engine = TradingEngine()
