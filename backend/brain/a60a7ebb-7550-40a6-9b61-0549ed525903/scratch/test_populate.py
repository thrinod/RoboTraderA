
import httpx
import asyncio

async def test_populate():
    url = "http://127.0.0.1:8000/scanner/populate_all"
    headers = {"X-App-Token": "thrinod"}
    
    print(f"Calling {url}...")
    try:
        async with httpx.AsyncClient(timeout=150.0) as client:
            # We don't know the password, but maybe the backend allows local requests?
            # Or maybe we can bypass it for this test if we run it on the same machine.
            # But the backend is likely checking the header.
            
            # Let's try without token first
            resp = await client.post(url, headers=headers)
            print(f"Status: {resp.status_code}")
            print(f"Body: {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_populate())
