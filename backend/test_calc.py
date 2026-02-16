import pandas as pd
import pandas_ta as ta
import numpy as np

def test_logic():
    print("Testing Logic...")
    # Create dummy DF
    dates = pd.date_range(start='2024-01-01', periods=200, freq='10T') # 200 rows, 10 min interval
    df = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 105,
        'low': np.random.randn(200).cumsum() + 95,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 200)
    }, index=dates)

    print("DF Created. Calculating BB...")
    try:
        # Bollinger Bands (20, 2)
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
             print("BB Calculated Columns:", bb.columns)
             df = pd.concat([df, bb], axis=1)
        else:
             print("BB returned None")

        # Pivot Points
        print("Calculating Pivots...")
        df_daily = df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        if len(df_daily) >= 1: # We might only have 1 day in dummy data
            print(f"Daily Rows: {len(df_daily)}")
            yesterday = df_daily.iloc[-1] # Use last for test
            yp_high = yesterday['high']
            yp_low = yesterday['low']
            yp_close = yesterday['close']
            
            pivot = (yp_high + yp_low + yp_close) / 3
            print(f"Pivot: {pivot}")
        
        print("Logic Test PASSED")
    except Exception as e:
        print(f"Logic Test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logic()
