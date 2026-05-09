# Agentic Trading AI Backend

This is a separate deployment application built parallel to the main RoboTrader backend. It is designed to use **FastAPI**, integrate with **Ollama** for local AI models, and host a multi-agent system (like `AnalysisAgent` and `TradingAgent`). It also provides an initial structure for **MCP (Model Context Protocol) Server Integration**.

## Features
- **FastAPI Backend**: Fast, asynchronous, and robust backend for the agentic workflows.
- **Ollama Integration**: Uses `core.llm_provider` to connect to a local or remote Ollama instance (`http://localhost:11434` by default).
- **Multi-Agent System**:
  - `AnalysisAgent`: Analyzes market data/symbols.
  - `TradingAgent`: Decides on actions (BUY, SELL, HOLD) based on the analysis.
  - `AgentCoordinator`: Coordinates the flow between agents.
- **MCP Server**: Exposes MCP tools via HTTP endpoints so other MCP-compatible clients can use this agentic AI to run analysis.

## Setup & Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Make sure Ollama is running (e.g., `ollama run llama3`).
3. Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## Endpoints
- `GET /` - Health check.
- `POST /api/trading/analyze` - Trigger the multi-agent workflow for a trading decision.
- `GET /mcp/tools` - List available MCP tools.
- `POST /mcp/tools/execute` - Execute an MCP tool (e.g., `analyze_asset`).
