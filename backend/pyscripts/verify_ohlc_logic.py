
import asyncio
import json

# Mock Response matching Upstox V3
mock_v3_response = {
    "status": "success",
    "data": {
        "NSE_FO|12345": {
            "ohlc": {
                "open": 100,
                "high": 120.5,
                "low": 90.5,
                "close": 110
            },
            "depth": {}
        },
        "NSE_FO|67890": {
            # Variant: Direct keys (Fallback test)
            "open": 200,
            "high": 250,
            "low": 180,
            "close": 210
        }
    }
}

async def test_parsing():
    flat_data = [
        {"instrument_key": "NSE_FO|12345", "strike": 24000},
        {"instrument_key": "NSE_FO|67890", "strike": 24100},
        {"instrument_key": "NSE_FO|MISSING", "strike": 24200},
    ]

    print("Testing Parsing Logic...")
    
    ohlc_map = {}
    
    # Simulate the loop logic from upstox_service.py
    ohlc_json = mock_v3_response
    if ohlc_json.get('status') == 'success' and 'data' in ohlc_json:
        for k, v in ohlc_json['data'].items():
            if 'ohlc' in v:
                ohlc_map[k] = v['ohlc']
            elif 'high' in v: 
                ohlc_map[k] = v

    print(f"OHLC Map: {ohlc_map}")

    # Simulate Merge
    for item in flat_data:
        ikey = item.get('instrument_key')
        if ikey and ikey in ohlc_map:
            ohlc = ohlc_map[ikey]
            item['high'] = ohlc.get('high', 0)
            item['low'] = ohlc.get('low', 0)
        else:
            item['high'] = 0
            item['low'] = 0
            
    print("Result Data:")
    print(json.dumps(flat_data, indent=2))

    # Assertions
    assert flat_data[0]['high'] == 120.5
    assert flat_data[1]['high'] == 250
    assert flat_data[2]['high'] == 0
    print("SUCCESS: Logic Verified")

if __name__ == "__main__":
    asyncio.run(test_parsing())
