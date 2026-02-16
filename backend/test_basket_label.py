import requests
import json

BASE_URL = "http://localhost:8000"

def test_basket_persistence():
    basket_id = 99
    payload = {
        "name": "Test Basket",
        "indexKey": "NSE_INDEX|Nifty 50",
        "slot1Key": "NSE_FO|12345",
        "slot1Label": "24000 CE Test",
        "slot2Key": "", "slot2Label": "",
        "slot3Key": "", "slot3Label": "",
        "slot4Key": "", "slot4Label": "",
        "indexInterval": "1minute",
        "slot1Interval": "1minute",
        "slot2Interval": "1minute",
        "slot3Interval": "1minute",
        "slot4Interval": "1minute"
    }

    print(f"Saving Basket {basket_id}...")
    try:
        res = requests.post(f"{BASE_URL}/analysis/basket/{basket_id}", json=payload)
        print("Save Status:", res.status_code, res.json())
    except Exception as e:
        print("Save Failed:", e)
        return

    print(f"Fetching Basket {basket_id}...")
    try:
        res = requests.get(f"{BASE_URL}/analysis/basket/{basket_id}")
        data = res.json()
        print("Fetch Status:", res.status_code)
        print("Fetched Label:", data.get("slot1Label"))
        
        if data.get("slot1Label") == "24000 CE Test":
            print("SUCCESS: Label persisted.")
        else:
            print("FAILURE: Label mismatch or missing.", data)
    except Exception as e:
        print("Fetch Failed:", e)

if __name__ == "__main__":
    test_basket_persistence()
