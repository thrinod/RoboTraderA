import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Agentic Trading AI Backend")

# Allow CORS for potential frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TradingRequest(BaseModel):
    symbol: str
    action: str
    context: str = ""

class RLFeedbackRequest(BaseModel):
    symbol: str
    action_taken: str
    outcome_reward: float
    lesson: str = ""

from agents.coordinator import AgentCoordinator
from mcp_integration.server import mcp_router
from core.logger import agent_logger

coordinator = AgentCoordinator()

app.include_router(mcp_router)

@app.get("/")
def read_root():
    return {"message": "Agentic Trading AI API is running. Powered by Ollama & Multi-Agents."}

@app.get("/api/logs")
def get_logs():
    return {"logs": agent_logger.get_logs()}

@app.post("/api/trading/analyze")
def analyze_trade(request: TradingRequest):
    # This endpoint will trigger the multi-agent workflow
    # integrating with Ollama to provide trading insights
    result = coordinator.process_trade_request(request.symbol, request.context)
    return {
        "status": "success",
        "symbol": result["symbol"],
        "proposed_decision": result["proposed_decision"],
        "critic_feedback": result["critic_feedback"],
        "final_decision": result["final_decision"],
        "agent_analysis": result["analysis"],
        "approved_by_critic": result["approved"]
    }

@app.post("/api/trading/rl-feedback")
def rl_feedback(request: RLFeedbackRequest):
    # This endpoint is used to train the RL Agent after a trade resolves
    coordinator.rl_agent.update_policy(
        symbol=request.symbol,
        action=request.action_taken,
        outcome_reward=request.outcome_reward,
        lesson=request.lesson
    )
    return {
        "status": "success",
        "message": f"RL policy updated for {request.symbol}"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
