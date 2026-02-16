import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_scanner():
    print("1. Fetching Instruments...")
    try:
        r = requests.get(f"{BASE_URL}/scanner/instruments")
        if r.status_code != 200:
            print(f"FAILED to fetch instruments: {r.status_code} {r.text}")
            return
        
        instruments = r.json().get('data', [])
        print(f"Found {len(instruments)} instruments in scanner DB.")
        
        if not instruments:
            print("No instruments to test processing.")
            return

        # key = instruments[0]['instrument_key']
        # limit to 3 keys for test
        test_keys = [i['instrument_key'] for i in instruments[:3]]
        print(f"Testing process for keys: {test_keys}")
        
        print("\n2. Calling /scanner/process (Batch)...")
        t0 = time.time()
        payload = {
            "instrument_keys": test_keys,
            "interval": "day"
        }
        r2 = requests.post(f"{BASE_URL}/scanner/process", json=payload)
        t1 = time.time()
        
        print(f"Response status: {r2.status_code}")
        print(f"Time taken: {t1 - t0:.2f}s")
        
        if r2.status_code == 200:
            data = r2.json().get('data', [])
            print(f"Received {len(data)} data items.")
            if data:
                item = data[0]
                print("\nSample Data Item Keys:")
                print(item.keys())
                print("\nIndicators Keys:")
                print(item.get('indicators', {}).keys())
                
                # Check for critical fields
                if 'ltp' in item and 'indicators' in item:
                    print("\nSUCCESS: Data structure appears valid.")
                else:
                    print("\nWARNING: Missing 'ltp' or 'indicators' in response.")
            else:
                print("Response data list is empty.")
        else:
            print(f"Error: {r2.text}")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    import sys
    # Redirect stdout to a file
    sys.stdout = open('debug_output.txt', 'w')
    test_scanner()
    sys.stdout.close()
