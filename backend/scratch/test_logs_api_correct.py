import requests
import json

def test_logs():
    dep_id = "69fde57946807337aa6b8dad"
    url = f"http://localhost:8000/deploy/logs/{dep_id}"
    headers = {"X-App-Token": "thrinod"}
    
    try:
        r = requests.get(url, headers=headers)
        print(f"Status: {r.status_code}")
        data = r.json().get('data', [])
        print(f"Count: {len(data)}")
        if data:
            print("First log sample:")
            print(json.dumps(data[0], indent=2))
        else:
            print(f"Response body: {r.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_logs()
