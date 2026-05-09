import numpy as np
# Monkey-patch NumPy 2.0 compatibility for pandas-ta
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "float_"):
    np.float_ = float

import pandas as pd
import pandas_ta as ta
import json

class BacktestService:
    def __init__(self, upstox_service):
        self.upstox_service = upstox_service

    async def _fetch_data(self, instrument_key: str, interval: str, days_back: int, use_intraday: bool):
        df = await self.upstox_service._fetch_historical_df(instrument_key, interval, days_back_override=days_back)
        if use_intraday:
            df_intra = await self.upstox_service._fetch_intraday_data_raw(instrument_key, interval)
            if df_intra is not None and not df_intra.empty:
                # Format Intraday logic
                cols = ['open', 'high', 'low', 'close', 'volume']
                for col in cols:
                    if col in df_intra.columns:
                        df_intra[col] = pd.to_numeric(df_intra[col])
                
                if 'timestamp' in df_intra.columns:
                    df_intra['timestamp'] = pd.to_datetime(df_intra['timestamp'])
                    if df_intra['timestamp'].dt.tz is not None:
                         df_intra['timestamp'] = df_intra['timestamp'].dt.tz_localize(None)
                    df_intra.set_index('timestamp', inplace=True)
                
                if df is None or df.empty:
                    df = df_intra
                else:
                    # Concatenate and Dedup
                    df = pd.concat([df, df_intra])
                    df = df[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)
        return df


    async def run_strategy(self, instrument_key: str, interval: str, days_back: int, stop_loss_pct: float = 1.0, take_profit_pct: float = 2.0, is_advanced: bool = False, execution_plan: list = None, saved_strategies: list = None, trade_type: str = "LONG", trade_instrument_key: str = None, use_intraday: bool = False):
        try:
            
            if not is_advanced or not execution_plan or not saved_strategies:
                # -------------------------------------------------------------
                # SIMPLE MODE (Hardcoded Strategy)
                # -------------------------------------------------------------
                df = await self._fetch_data(instrument_key, interval, days_back, use_intraday)
                if df is None or df.empty or len(df) < 50:
                    return {"status": "error", "message": "Insufficient historical data for backtesting."}
                
                df.ta.bbands(length=20, std=2, append=True)
                df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
                df.ta.macd(fast=12, slow=26, signal=9, append=True)
                
                bbl_col = [c for c in df.columns if c.startswith('BBL_')][0]
                stoch_k_col = [c for c in df.columns if c.startswith('STOCHk_')][0]
                stoch_d_col = [c for c in df.columns if c.startswith('STOCHd_')][0]
                macd_h_col = [c for c in df.columns if c.startswith('MACDh_')][0]

                cond1_bb_touch = df['low'].shift(1) <= df[bbl_col].shift(1)
                cond2_stoch_oversold = df[stoch_k_col].shift(1) < 20
                cond3_stoch_cross = (df[stoch_k_col].shift(1) <= df[stoch_d_col].shift(1)) & \
                                    (df[stoch_k_col] > df[stoch_d_col])
                cond4_macd_momentum = df[macd_h_col] > df[macd_h_col].shift(1)

                df['Buy_Signal'] = cond1_bb_touch & cond2_stoch_oversold & cond3_stoch_cross & cond4_macd_momentum
                df['Buy_Signal'] = df['Buy_Signal'].fillna(False).astype(int)

                condition_stats = {
                    "BB_Touch_Hits": int(cond1_bb_touch.sum()),
                    "Stoch_Oversold_Hits": int(cond2_stoch_oversold.sum()),
                    "Stoch_Cross_Hits": int(cond3_stoch_cross.sum()),
                    "MACD_Momentum_Hits": int(cond4_macd_momentum.sum()),
                    "Total_Combined_Hits": int(df['Buy_Signal'].sum()),
                    "Total_Candles_Analyzed": len(df)
                }
                
                master_df = df
            else:
                # -------------------------------------------------------------
                # ADVANCED MODE (Dynamic Multi-Leg Evaluator)
                # -------------------------------------------------------------
                leg_dfs = {}
                leg_signals = {}
                condition_stats = {}
                
                def get_series(t_df, name):
                    is_prev = name.endswith('_prev')
                    base_name = name.replace('_prev', '')
                    
                    if base_name == '0':
                        s = pd.Series(0, index=t_df.index)
                    else:
                        col_map = {
                            'close': 'close', 'open': 'open', 'low': 'low', 'high': 'high',
                            'STOCHk': next((c for c in t_df.columns if c.startswith('STOCHk_')), 'close'),
                            'STOCHd': next((c for c in t_df.columns if c.startswith('STOCHd_')), 'close'),
                            'MACDh': next((c for c in t_df.columns if c.startswith('MACDh_')), 'close'),
                            'BBL_20_2': next((c for c in t_df.columns if c.startswith('BBL_')), 'close'),
                            'BBU_20_2': next((c for c in t_df.columns if c.startswith('BBU_')), 'close'),
                            'STOCHRSIk': next((c for c in t_df.columns if c.startswith('STOCHRSIk_')), 'close'),
                            'STOCHRSId': next((c for c in t_df.columns if c.startswith('STOCHRSId_')), 'close'),
                        }
                        s = t_df[col_map.get(base_name, 'close')]
                        
                    if is_prev:
                        s = s.shift(1)
                    return s
                
                for idx, leg in enumerate(execution_plan):
                    leg_id = leg['id']
                    ik = leg['leg']
                    tf = leg['timeframe']
                    strat_id = leg['strategyId']
                    
                    df_leg = await self._fetch_data(ik, tf, days_back, use_intraday)
                    if df_leg is None or df_leg.empty:
                        return {"status": "error", "message": f"Failed to fetch data for Leg {idx + 1} ({ik}). Please ensure your Upstox session is active and the instrument key is valid."}
                    
                    if len(df_leg) < 30:
                        return {"status": "error", "message": f"Insufficient history for Leg {idx + 1} ({ik}). Found only {len(df_leg)} candles. Need at least 30."}
                        
                    # Pre-compute all possible indicators for this leg
                    try:
                        df_leg.ta.bbands(length=20, std=2, append=True)
                        df_leg.ta.stoch(k=14, d=3, smooth_k=3, append=True)
                        df_leg.ta.macd(fast=12, slow=26, signal=9, append=True)
                        df_leg.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
                    except Exception as e:
                        print(f"Indicator calc error on {ik}: {e}")
                    
                    leg_dfs[leg_id] = df_leg
                    leg_match = pd.Series(True, index=df_leg.index)
                    
                    # Find and evaluate strategy rules
                    strat = next((s for s in saved_strategies if s['id'] == str(strat_id)), None)
                    if strat and 'rules' in strat:
                        for rule in strat['rules']:
                            try:
                                left = get_series(df_leg, rule['indicator'])
                                right = float(rule['value']) if rule['valueType'] == 'number' else get_series(df_leg, rule['value'])
                                op = rule['operator']
                                
                                if op == '>': rule_match = left > right
                                elif op == '<': rule_match = left < right
                                elif op == '>=': rule_match = left >= right
                                elif op == '<=': rule_match = left <= right
                                elif op == '==': rule_match = left == right
                                elif op == 'cross_above': rule_match = (left > right) & (left.shift(1) <= right.shift(1))
                                elif op == 'cross_below': rule_match = (left < right) & (left.shift(1) >= right.shift(1))
                                else: rule_match = pd.Series(True, index=df_leg.index)
                                
                                leg_match = leg_match & rule_match
                            except Exception as e:
                                print(f"Rule eval error: {e}")
                                
                    leg_signals[leg_id] = leg_match
                    condition_stats[f"Leg {idx+1} ({ik} - {tf}) Hits"] = int(leg_match.sum())
                    
                # Synchronize legs onto the master timeframe (Leg 1)
                master_leg_id = execution_plan[0]['id']
                if master_leg_id not in leg_dfs:
                    return {"status": "error", "message": "Failed to fetch data for Primary Leg."}
                    
                master_df = leg_dfs[master_leg_id].copy()
                master_signal = pd.Series(True, index=master_df.index)
                
                for leg in execution_plan:
                    leg_id = leg['id']
                    if leg_id in leg_signals:
                        aligned = leg_signals[leg_id].reindex(master_df.index, method='ffill').fillna(False)
                        master_signal = master_signal & aligned
                        
                master_df['Buy_Signal'] = master_signal.astype(int)

            # -------------------------------------------------------------
            # TRADE INSTRUMENT ALIGNMENT
            # -------------------------------------------------------------
            # If a separate trade instrument is provided, we fetch its data and align it
            # with our master_df (the signal source).
            if trade_instrument_key and trade_instrument_key != instrument_key:
                print(f"Aligning trade instrument: {trade_instrument_key}")
                trade_df = await self._fetch_data(trade_instrument_key, interval, days_back, use_intraday)
                if trade_df is not None and not trade_df.empty:
                    # Rename columns to avoid collision and align
                    trade_df = trade_df[['open', 'high', 'low', 'close']].copy()
                    trade_df.columns = ['t_open', 't_high', 't_low', 't_close']
                    
                    # Align trade data to master signal timeline
                    master_df = master_df.join(trade_df, how='left')
                    # Forward fill missing trade prices (e.g. if trade instrument is less liquid)
                    master_df[['t_open', 't_high', 't_low', 't_close']] = master_df[['t_open', 't_high', 't_low', 't_close']].ffill()
                else:
                    print(f"Warning: Failed to fetch trade instrument {trade_instrument_key}")
            else:
                # Default: Use same instrument for trading
                master_df['t_open'] = master_df['open']
                master_df['t_high'] = master_df['high']
                master_df['t_low'] = master_df['low']
                master_df['t_close'] = master_df['close']
                condition_stats["Total_Combined_Hits"] = int(master_df['Buy_Signal'].sum())
                condition_stats["Total_Candles_Analyzed"] = len(master_df)

            # 4. Simulate Trades
            trades = []
            signals_log = []
            in_position = False
            entry_price = 0
            entry_time = None
            
            # Compounding Capital Logic
            current_capital = 100000.0
            peak_capital = current_capital
            max_drawdown_pct = 0.0
            
            for i in range(1, len(master_df)):
                # Log the raw signal if it fired
                if master_df['Buy_Signal'].iloc[i] == 1:
                    is_traded = not in_position
                    ts_val = master_df.index[i]
                    ts_str = ts_val.isoformat() if hasattr(ts_val, 'isoformat') else str(ts_val)
                    
                    # For advanced mode, include detailed leg-level indicator values
                    leg_details = []
                    if is_advanced and execution_plan:
                        for leg in execution_plan:
                            lid = leg['id']
                            if lid in leg_dfs:
                                try:
                                    leg_df = leg_dfs[lid]
                                    # Find the most recent candle for this leg relative to master timestamp
                                    # (Handles cross-timeframe alignment)
                                    idx = leg_df.index.searchsorted(ts_val, side='right') - 1
                                    if idx >= 0:
                                        row = leg_df.iloc[idx]
                                        # Extract TA indicators (non-core columns)
                                        leg_indicators = {
                                            col: round(float(row[col]), 4) if isinstance(row[col], (int, float)) else str(row[col])
                                            for col in leg_df.columns 
                                            if col not in ['open', 'high', 'low', 'close', 'volume', 'oi', 'Buy_Signal', 'timestamp']
                                        }
                                        leg_details.append({
                                            "instrument": leg['leg'],
                                            "timeframe": leg['timeframe'],
                                            "indicators": leg_indicators,
                                            "matched": bool(leg_signals[lid].iloc[idx]) if lid in leg_signals else True
                                        })
                                except Exception as e:
                                    print(f"Detail log error: {e}")

                    signals_log.append({
                        "timestamp": ts_str,
                        "price": float(master_df['t_close'].iloc[i]),
                        "traded": is_traded,
                        "reason": f"Executed {trade_type}" if is_traded else "Skipped (In position)",
                        "leg_details": leg_details if is_advanced else None,
                        "indicators": {
                            "Status": "All Legs Aligned" if not leg_details else f"{len(leg_details)} Legs Evaluated"
                        }
                    })

                if not in_position and master_df['Buy_Signal'].iloc[i] == 1:
                    in_position = True
                    entry_price = float(master_df['t_close'].iloc[i])
                    entry_time = master_df.index[i]
                    continue
                    
                if in_position:
                    current_low = float(master_df['t_low'].iloc[i])
                    current_high = float(master_df['t_high'].iloc[i])
                    current_close = float(master_df['t_close'].iloc[i])

                    if trade_type == "SHORT":
                        low_pnl_pct = (entry_price - current_high) / entry_price * 100
                        high_pnl_pct = (entry_price - current_low) / entry_price * 100
                    else:
                        low_pnl_pct = (current_low - entry_price) / entry_price * 100
                        high_pnl_pct = (current_high - entry_price) / entry_price * 100
                    
                    hit_sl = low_pnl_pct <= -stop_loss_pct
                    hit_tp = high_pnl_pct >= take_profit_pct
                    
                    exit_cond_met = False
                    
                    if hit_sl:
                        if trade_type == "SHORT":
                            exit_price = entry_price * (1 + stop_loss_pct / 100)
                        else:
                            exit_price = entry_price * (1 - stop_loss_pct / 100)
                        pnl_pct = -stop_loss_pct
                        exit_cond_met = True
                    elif hit_tp:
                        if trade_type == "SHORT":
                            exit_price = entry_price * (1 - take_profit_pct / 100)
                        else:
                            exit_price = entry_price * (1 + take_profit_pct / 100)
                        pnl_pct = take_profit_pct
                        exit_cond_met = True
                    elif i == len(master_df) - 1:
                        exit_price = current_close
                        if trade_type == "SHORT":
                            pnl_pct = (entry_price - current_close) / entry_price * 100
                        else:
                            pnl_pct = (current_close - entry_price) / entry_price * 100
                        exit_cond_met = True

                    if exit_cond_met:
                        exit_time = master_df.index[i]
                        
                        # Compounding Calculation
                        capital_change = current_capital * (pnl_pct / 100.0)
                        current_capital += capital_change
                        
                        # Drawdown tracking
                        if current_capital > peak_capital:
                            peak_capital = current_capital
                        drawdown = (peak_capital - current_capital) / peak_capital * 100
                        if drawdown > max_drawdown_pct:
                            max_drawdown_pct = drawdown
                            
                        trades.append({
                            "entry_time": entry_time.isoformat(),
                            "entry_price": round(entry_price, 2),
                            "exit_time": exit_time.isoformat(),
                            "exit_price": round(exit_price, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "pnl_value": round(capital_change, 2),
                            "running_capital": round(current_capital, 2),
                            "is_win": pnl_pct > 0
                        })
                        in_position = False

            win_trades = [t for t in trades if t['is_win']]
            win_rate = (len(win_trades) / len(trades) * 100) if trades else 0
            
            # Attach equity curve to chart data if needed, but UI can build it from trades.
            display_df = master_df.tail(500).copy()
            chart_json = display_df.reset_index().assign(timestamp=lambda x: x['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S+05:30')).to_json(orient="records")
            chart_data = json.loads(chart_json)

            return {
                "status": "success",
                "summary": {
                    "total_trades": len(trades),
                    "win_rate": round(win_rate, 2),
                    "total_pnl": round(current_capital - 100000.0, 2),
                    "final_capital": round(current_capital, 2),
                    "max_drawdown_pct": round(max_drawdown_pct, 2),
                    "avg_pnl_pct": round(sum(t['pnl_pct'] for t in trades) / len(trades), 2) if trades else 0,
                    "condition_stats": condition_stats
                },
                "trades": trades,
                "signals": signals_log,
                "chart_data": chart_data
            }
            
        except Exception as e:
            import traceback
            print(f"Backtest error: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
