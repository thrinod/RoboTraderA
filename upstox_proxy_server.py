from fastapi import FastAPI, Request, Response
import httpx
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("upstox-proxy")

app = FastAPI(title="Upstox Proxy Server")

UPSTOX_BASE_URL = "https://api.upstox.com"

# Reusable client to keep connections alive, forcing IPv4 via binding to 0.0.0.0
transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0")
client = httpx.AsyncClient(base_url=UPSTOX_BASE_URL, timeout=30.0, transport=transport)

@app.on_event("shutdown")
async def shutdown():
    await client.aclose()

@app.get("/")
async def ping():
    """Health check endpoint to verify proxy status."""
    return {"status": "active", "message": "Upstox Proxy Server is running"}

@app.get("/check-ip")
async def check_proxy_ip():
    """Returns the public IP of this proxy server to confirm outgoing traffic origin."""
    try:
        resp = await client.get("https://api.ipify.org?format=json")
        return {"proxy_server_ip": resp.json().get("ip"), "status": "Success. Upstox sees this IP."}
    except Exception as e:
        return {"error": str(e)}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    client_ip = request.client.host if request.client else "Unknown"
    logger.info(f"[{client_ip}] -> Proxying {request.method} to Upstox: /{path}")
    
    # 1. Forward headers (excluding Host which needs to be upstox)
    headers = dict(request.headers)
    headers.pop("host", None)
    
    # 2. Extract body
    body = await request.body()
    
    # 3. Extract query params
    params = dict(request.query_params)
    
    # 4. Make request to Upstox
    try:
        upstox_resp = await client.request(
            method=request.method,
            url=f"/{path}",
            headers=headers,
            params=params,
            content=body
        )
        
        logger.info(f"[{client_ip}] <- Received {upstox_resp.status_code} from Upstox for /{path}")
        
        # 5. Return Upstox's response exactly as it is
        return Response(
            content=upstox_resp.content,
            status_code=upstox_resp.status_code,
            headers=dict(upstox_resp.headers)
        )
    except Exception as e:
        logger.error(f"Error proxying to Upstox: {e}")
        return Response(content=str(e), status_code=502)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
