from fastapi import APIRouter

mcp_router = APIRouter()

# MCP (Model Context Protocol) Integration
# This allows external tools or other agents to connect to this AI trading backend
# using the MCP standard (e.g., exposing tools, prompts, or resources)

@mcp_router.get("/mcp/tools")
def list_tools():
    """List available MCP tools from this server"""
    return {
        "tools": [
            {
                "name": "analyze_asset",
                "description": "Analyze an asset using multi-agent workflow",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock or crypto symbol (e.g., AAPL, BTC/USD)"
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context for the analysis"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        ]
    }

@mcp_router.post("/mcp/tools/execute")
def execute_tool(payload: dict):
    """Execute an MCP tool"""
    from agents.coordinator import AgentCoordinator
    
    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments", {})
    
    if tool_name == "analyze_asset":
        coordinator = AgentCoordinator()
        result = coordinator.process_trade_request(
            symbol=arguments.get("symbol"),
            context=arguments.get("context", "")
        )
        return {"result": result}
    
    return {"error": "Tool not found"}
