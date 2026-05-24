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
        self.loop_task = None
        self.symbol_cache = {} # Cache for instrument_key -> trading_symbol mapping

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

        # Check Global Settings Constraints
        try:
            general_config = await self.mongodb["settings"].find_one({"id": "general_config"})
            if general_config:
                # Check Pause All
                if general_config.get("pause_all_deployments", False):
                    return  # Silently skip all evaluations when paused
                # Check Trading Hours
                if general_config.get("enforce_market_hours", False):
                    now = datetime.now().time()
                    import datetime as dt
                    market_start = dt.time(9, 0)
                    market_end = dt.time(15, 30)
                    if not (market_start <= now <= market_end):
                        return
        except Exception as e:
            print(f"Error checking global config: {e}")

        # 1. Fetch Active + Monitoring Deployments
        try:
            cursor = self.mongodb["strategy_deployments"].find({"status": {"$in": ["ACTIVE", "MONITORING"]}})
            deployments = await cursor.to_list(length=100)
        except Exception as e:
            print(f"Error fetching deployments: {e}")
            return

        active_deps = [d for d in deployments if d.get('status') == 'ACTIVE']
        monitoring_deps = [d for d in deployments if d.get('status') == 'MONITORING']

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

        async def _safe_monitor(dep):
            async with sem:
                try:
                    await asyncio.wait_for(self.monitor_position(dep), timeout=15.0)
                except asyncio.TimeoutError:
                    print(f"TIMEOUT monitoring deployment {dep.get('_id')}")
                except Exception as e:
                    print(f"Error monitoring deployment {dep.get('_id')}: {e}")

        tasks = [_safe_evaluate(dep) for dep in active_deps]
        tasks += [_safe_monitor(dep) for dep in monitoring_deps]
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

        # 1. Fetch Latest Data (Two-stage: Intraday first, then Historical fallback)
        df = None
        try:
            # Stage 1: Try intraday API (works reliably for F&O and current day)
            if interval != "day":
                df_intraday = await upstox_service._fetch_intraday_df(ik, interval)
            else:
                df_intraday = pd.DataFrame()

            # Stage 2: Fetch historical data for indicator lookback
            df_historical = await upstox_service._fetch_historical_df(ik, interval, days_back_override=5)

            # Merge: combine historical + intraday, deduplicate by timestamp
            frames = []
            if df_historical is not None and not df_historical.empty:
                frames.append(df_historical)
            if df_intraday is not None and not df_intraday.empty:
                frames.append(df_intraday)

            if frames:
                df = pd.concat(frames)
                df = df[~df.index.duplicated(keep='last')]  # Keep latest data on overlap
                df.sort_index(inplace=True)
        except Exception as e:
            raise e

        # F&O instruments may have fewer candles; use a lower threshold
        is_fno = ik.startswith("NSE_FO") or ik.startswith("BSE_FO")
        min_candles = 5 if is_fno else 20

        if df is None or df.empty or len(df) < min_candles:
            candle_count = 0 if df is None else len(df)
            error_msg = f"Data fetch failed for {ik}: got {candle_count} candles (need {min_candles}+). Check Upstox token."
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

        # 2. Calculate Indicators (resilient to low candle counts)
        try:
            if len(df) >= 20:
                df.ta.bbands(length=20, std=2, append=True)
            if len(df) >= 14:
                df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
            if len(df) >= 26:
                df.ta.macd(fast=12, slow=26, signal=9, append=True)
            if len(df) >= 14:
                df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
        except Exception as e:
            print(f"Indicator calc warning for {ik}: {e}")

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
                    resolved_symbol = await self.get_symbol_info(trade_instrument_key)
                    msg = (
                        f"⚠️ <b>Trade Skipped</b>\n\n"
                        f"<b>Instrument:</b> {resolved_symbol}\n"
                        f"<b>Key:</b> {trade_instrument_key}\n"
                        f"Allocated Capital (₹{calculated_capital:.2f}) is too low to buy 1 lot. Cost/Lot: ₹{cost_per_lot:.2f}"
                    )
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
            
        # Calculate SL/TP price levels
        stop_loss_pct = dep.get('stop_loss', 0)
        take_profit_pct = dep.get('take_profit', 0)
        direction = dep.get('trade_type', 'LONG')

        sl_price = 0
        tp_price = 0
        if direction == 'LONG':
            if stop_loss_pct > 0:
                sl_price = round(trade_price * (1 - stop_loss_pct / 100), 2)
            if take_profit_pct > 0:
                tp_price = round(trade_price * (1 + take_profit_pct / 100), 2)
        else:  # SHORT
            if stop_loss_pct > 0:
                sl_price = round(trade_price * (1 + stop_loss_pct / 100), 2)
            if take_profit_pct > 0:
                tp_price = round(trade_price * (1 - take_profit_pct / 100), 2)

        # Update deployment: move to MONITORING with entry details
        update_fields = {
            "status": "MONITORING",
            "last_trade_at": datetime.now(),
            "entry_price": trade_price,
            "entry_qty": qty,
            "entry_side": side,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "trade_instrument_key_actual": trade_instrument_key,
        }
        await self.mongodb["strategy_deployments"].update_one(
            {"_id": dep["_id"]},
            {"$set": update_fields}
        )

        # Telegram Notification
        symbol_info = await self.get_symbol_info(trade_instrument_key)
        option_info = await telegram_service.get_option_display_info(trade_instrument_key)
        option_block = f"{option_info}" if option_info else ""
        sl_info = f"\n<b>Stop Loss:</b> ₹{sl_price} ({stop_loss_pct}%)" if sl_price > 0 else ""
        tp_info = f"\n<b>Take Profit:</b> ₹{tp_price} ({take_profit_pct}%)" if tp_price > 0 else ""
        msg = (
            f"🚀 <b>Trade Executed!</b>\n\n"
            f"<b>Instrument:</b> {symbol_info}\n"
            f"<b>Key:</b> {trade_instrument_key}\n"
            f"{option_block}"
            f"<b>Side:</b> {side}\n"
            f"<b>Quantity:</b> {qty}\n"
            f"<b>Price:</b> ₹{trade_price}\n"
            f"<b>Mode:</b> {mode}"
            f"{sl_info}{tp_info}\n"
            f"<b>Status:</b> Now monitoring for SL/TP\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await telegram_service.send_message(msg)

    async def monitor_position(self, dep):
        """Monitor an open position for SL/TP hit."""
        trade_ik = dep.get('trade_instrument_key_actual') or dep.get('trade_instrument_key') or dep.get('primary_instrument')
        entry_price = dep.get('entry_price', 0)
        sl_price = dep.get('sl_price', 0)
        tp_price = dep.get('tp_price', 0)
        direction = dep.get('trade_type', 'LONG')
        mode = dep.get('deployment_mode', 'MOCK')
        qty = dep.get('entry_qty', dep.get('quantity', 1))

        if not trade_ik or entry_price <= 0:
            return

        # If both SL and TP are 0, nothing to monitor — mark as TRADED
        if sl_price <= 0 and tp_price <= 0:
            await self.mongodb["strategy_deployments"].update_one(
                {"_id": dep["_id"]},
                {"$set": {"status": "TRADED"}}
            )
            return

        # Fetch live price
        try:
            quotes = upstox_service.get_market_quotes([trade_ik])
            ltp = quotes.get(trade_ik, {}).get('ltp', 0)
        except Exception as e:
            print(f"Monitor: Failed to fetch LTP for {trade_ik}: {e}")
            return

        if ltp <= 0:
            return

        # Calculate unrealized P&L
        if direction == 'LONG':
            pnl = round((ltp - entry_price) * qty, 2)
            profit_pct = ((ltp - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        else:
            pnl = round((entry_price - ltp) * qty, 2)
            profit_pct = ((entry_price - ltp) / entry_price) * 100 if entry_price > 0 else 0

        # Trailing SL Logic: Move to Break-even
        trailing_sl_enabled = dep.get('trailing_sl', False)
        trailing_trigger = dep.get('trailing_sl_trigger_pct', 40.0)
        sl_trailed = dep.get('sl_trailed', False)

        if trailing_sl_enabled and not sl_trailed and profit_pct >= trailing_trigger:
            print(f"Monitor: Profit reached {profit_pct:.2f}%, trailing SL to Break-even (₹{entry_price})")
            sl_price = entry_price
            await self.mongodb["strategy_deployments"].update_one(
                {"_id": dep["_id"]},
                {"$set": {
                    "sl_price": sl_price,
                    "sl_trailed": True,
                    "sl_trailed_at": datetime.now()
                }}
            )
            # Update local dep object so subsequent logic uses new sl_price
            dep['sl_price'] = sl_price
            dep['sl_trailed'] = True

            # Notify via Telegram
            symbol_info = await self.get_symbol_info(trade_ik)
            option_info = await telegram_service.get_option_display_info(trade_ik)
            option_block = f"{option_info}" if option_info else ""
            msg = (
                f"🛡️ <b>Trailing SL Activated!</b>\n\n"
                f"<b>Instrument:</b> {symbol_info}\n"
                f"<b>Key:</b> {trade_ik}\n"
                f"{option_block}"
                f"<b>Profit:</b> {profit_pct:.2f}%\n"
                f"<b>New SL:</b> ₹{sl_price} (Break-even)\n"
                f"<b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            )
            try:
                await telegram_service.send_message(msg)
            except: pass

        # Always update the deployment doc with latest LTP + P&L for UI display
        await self.mongodb["strategy_deployments"].update_one(
            {"_id": dep["_id"]},
            {"$set": {
                "live_ltp": ltp,
                "live_pnl": pnl,
                "ltp_updated_at": datetime.now()
            }}
        )

        # Check SL/TP conditions
        sl_hit = False
        tp_hit = False
        exit_reason = ""

        if direction == 'LONG':
            if sl_price > 0 and ltp <= sl_price:
                sl_hit = True
                exit_reason = "STOP_LOSS"
            elif tp_price > 0 and ltp >= tp_price:
                tp_hit = True
                exit_reason = "TAKE_PROFIT"
        else:  # SHORT
            if sl_price > 0 and ltp >= sl_price:
                sl_hit = True
                exit_reason = "STOP_LOSS"
            elif tp_price > 0 and ltp <= tp_price:
                tp_hit = True
                exit_reason = "TAKE_PROFIT"

        if sl_hit or tp_hit:
            # Execute exit trade
            await self.execute_exit(dep, trade_ik, ltp, qty, exit_reason, pnl, mode)
        else:
            # Log monitoring check (throttled — only log every ~30s to avoid spam)
            # Check last log timestamp for this deployment
            last_log = await self.mongodb["deployment_logs"].find_one(
                {"deployment_id": str(dep["_id"]), "type": "MONITOR"},
                sort=[("timestamp", -1)]
            )
            should_log = True
            if last_log and last_log.get('timestamp'):
                elapsed = (datetime.now() - last_log['timestamp']).total_seconds()
                should_log = elapsed >= 30  # Log every 30 seconds

            if should_log:
                await self.mongodb["deployment_logs"].insert_one({
                    "deployment_id": str(dep["_id"]),
                    "timestamp": datetime.now(),
                    "type": "MONITOR",
                    "instrument": trade_ik,
                    "entry_price": entry_price,
                    "ltp": ltp,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "pnl": pnl,
                    "signal": False,
                    "traded": False,
                    "rules": [
                        {"rule": f"SL Check (₹{sl_price})", "matched": sl_hit, "left": ltp, "right": sl_price} if sl_price > 0 else None,
                        {"rule": f"TP Check (₹{tp_price})", "matched": tp_hit, "left": ltp, "right": tp_price} if tp_price > 0 else None,
                    ],
                    "message": f"Monitoring: LTP=₹{ltp} | Entry=₹{entry_price} | P&L=₹{pnl} | SL=₹{sl_price} | TP=₹{tp_price}"
                })

    async def execute_exit(self, dep, trade_ik, exit_price, qty, reason, pnl, mode):
        """Execute exit trade when SL or TP is hit."""
        exit_side = "SELL" if dep.get('trade_type', 'LONG') == 'LONG' else "BUY"
        entry_price = dep.get('entry_price', 0)

        print(f"Engine: EXIT {exit_side} {qty} {trade_ik} at {exit_price} ({mode}) — {reason}")

        try:
            if mode == "MOCK":
                # Find and exit the mock position
                mock_trade = await self.mongodb["mock_trades"].find_one({
                    "instrument_key": trade_ik,
                    "status": "OPEN"
                })
                if mock_trade:
                    await mock_trade_service.exit_position(mock_trade['trade_id'])
                else:
                    # Place a counter-order
                    await mock_trade_service.place_order({
                        "instrument_key": trade_ik,
                        "quantity": qty,
                        "transaction_type": exit_side,
                        "price": exit_price
                    })
            else:
                # LIVE exit via Upstox
                await upstox_service.place_order(
                    instrument_key=trade_ik,
                    quantity=qty,
                    transaction_type=exit_side,
                    order_type="MARKET"
                )
        except Exception as e:
            print(f"Engine: Exit order failed: {e}")

        # Update deployment status
        await self.mongodb["strategy_deployments"].update_one(
            {"_id": dep["_id"]},
            {"$set": {
                "status": "EXITED",
                "exit_price": exit_price,
                "exit_reason": reason,
                "exit_pnl": pnl,
                "exited_at": datetime.now()
            }}
        )

        # Log the exit event
        await self.mongodb["deployment_logs"].insert_one({
            "deployment_id": str(dep["_id"]),
            "timestamp": datetime.now(),
            "type": "EXIT",
            "instrument": trade_ik,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
            "signal": True,
            "traded": True,
            "rules": [{"rule": reason, "matched": True, "left": exit_price, "right": dep.get('sl_price' if reason == 'STOP_LOSS' else 'tp_price', 0)}],
            "message": f"{'🛑 STOP LOSS' if reason == 'STOP_LOSS' else '🎯 TAKE PROFIT'} HIT — Exited {exit_side} {qty} at ₹{exit_price} | P&L: ₹{pnl}"
        })

        # Telegram Notification
        symbol_info = await self.get_symbol_info(trade_ik)
        option_info = await telegram_service.get_option_display_info(trade_ik)
        option_block = f"{option_info}" if option_info else ""
        emoji = "🛑" if reason == "STOP_LOSS" else "🎯"
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{emoji} <b>Position Closed — {reason.replace('_', ' ')}</b>\n\n"
            f"<b>Instrument:</b> {symbol_info}\n"
            f"<b>Key:</b> {trade_ik}\n"
            f"{option_block}"
            f"<b>Entry:</b> ₹{entry_price}\n"
            f"<b>Exit:</b> ₹{exit_price}\n"
            f"<b>Qty:</b> {qty}\n"
            f"{pnl_emoji} <b>P&L:</b> ₹{pnl}\n"
            f"<b>Mode:</b> {mode}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        try:
            await telegram_service.send_message(msg)
        except: pass

    async def get_symbol_info(self, instrument_key):
        """Helper to get a human readable symbol from instrument key"""
        if not instrument_key:
            return "Unknown"
        
        if instrument_key in self.symbol_cache:
            return self.symbol_cache[instrument_key]
        
        if self.mongodb is not None:
            try:
                # Try scanner_instruments_main first (user selected ones)
                instr = await self.mongodb["scanner_instruments_main"].find_one({"instrument_key": instrument_key})
                if not instr:
                    # Try upstox_collection (master list)
                    instr = await self.mongodb["upstox_collection"].find_one({"instrument_key": instrument_key})
                
                if instr:
                    symbol = instr.get('trading_symbol') or instr.get('name')
                    name = instr.get('name')
                    if symbol:
                        if name and name.upper() != symbol.upper():
                            display_name = f"{symbol} ({name})"
                        else:
                            display_name = symbol
                        self.symbol_cache[instrument_key] = display_name
                        return display_name
            except Exception as e:
                print(f"Error fetching symbol for {instrument_key}: {e}")
        
        # Fallback to key if symbol not found (strip key prefix if present with explanatory label)
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

trading_engine = TradingEngine()
