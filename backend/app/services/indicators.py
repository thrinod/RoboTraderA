import pandas as pd
import pandas_ta as ta

class Indicators:
    @staticmethod
    def calculate_all(df: pd.DataFrame):
        """
        Calculates all required indicators and appends them to the DataFrame.
        Expected generic columns: 'Open', 'High', 'Low', 'Close', 'Volume'.
        """
        # Ensure capitalization matches pandas-ta requirements if needed, usually it's case insensitive or standard Caps.
        # standardizing columns
        df.columns = [c.capitalize() for c in df.columns]

        # 1. Bollinger Bands
        # Returns: BBL (Lower), BBM (Mid), BBU (Upper), BBB (Bandwidth), BBP (%B)
        df.ta.bbands(append=True)

        # 2. RSI (Relative Strength Index)
        df.ta.rsi(length=14, append=True)

        # 3. MACD
        # Returns: MACD, MACDh (Hist), MACDs (Signal)
        df.ta.macd(append=True)

        # 4. Moving Averages
        df.ta.sma(length=20, append=True) # SMA 20
        df.ta.ema(length=50, append=True) # EMA 50

        # 5. DMI (Directional Movement Index) / ADX
        # Returns: ADX, DMP (+DI), DMN (-DI)
        df.ta.adx(append=True)

        # 6. Pivot Points (Standard)
        # Note: Pivot points are usually calculated on the PREVIOUS period's data for the current period.
        # pandas-ta pivot implementation might differ, let's check standard. 
        # For simplicity, we can implement standard pivot manually or use ta.
        # df.ta.pivots(append=True) # usage might vary based on version, manual fallback is safer for basic standard.
        
        # Manual Pivot Calculation for last completed candle
        last_row = df.iloc[-1]
        high = last_row['High']
        low = last_row['Low']
        close = last_row['Close']
        pivot = (high + low + close) / 3
        df['Pivot_P'] = pivot
        df['Pivot_R1'] = (2 * pivot) - low
        df['Pivot_S1'] = (2 * pivot) - high
        
        df['Pivot_R2'] = pivot + (high - low)
        df['Pivot_S2'] = pivot - (high - low)
        
        df['Pivot_R3'] = high + 2 * (pivot - low)
        df['Pivot_S3'] = low - 2 * (high - pivot)
        
        return df

indicators = Indicators()
