from agents.analysis_agent import AnalysisAgent
from agents.trading_agent import TradingAgent
from agents.critic_agent import CriticAgent
from agents.rl_agent import RLAgent
from mcp_integration.client import mcp_client_manager
from core.logger import agent_logger

class AgentCoordinator:
    def __init__(self):
        self.analysis_agent = AnalysisAgent()
        self.trading_agent = TradingAgent()
        self.critic_agent = CriticAgent()
        self.rl_agent = RLAgent()
        self.mcp_manager = mcp_client_manager

    def process_trade_request(self, symbol: str, context: str):
        agent_logger.log("Coordinator", "Workflow Started", f"Starting workflow for {symbol}")
        
        # Fetch available remote MCP servers to pass as context
        available_servers = self.mcp_manager.list_configured_servers()
        
        # Step 1: Get Reinforcement Learning historical context
        rl_context = self.rl_agent.get_learned_context(symbol)
        
        enriched_context = f"{context}\nAvailable trading tools: {', '.join(available_servers)}\n{rl_context}"
        
        # Step 2: Analyze
        analysis_result = self.analysis_agent.analyze(symbol, enriched_context)
        
        # Step 3: Decide (Proposed Action)
        proposed_decision = self.trading_agent.decide(symbol, analysis_result)
        
        # Step 4: Critique
        critique = self.critic_agent.critique(symbol, analysis_result, proposed_decision, rl_context)
        
        final_decision = proposed_decision if critique["approved"] else f"HOLD (Rejected by Critic: {critique['feedback']})"
        
        agent_logger.log("Coordinator", "Workflow Complete", f"Final decision for {symbol}: {final_decision}")
        
        return {
            "symbol": symbol,
            "analysis": analysis_result,
            "proposed_decision": proposed_decision,
            "critic_feedback": critique["feedback"],
            "final_decision": final_decision,
            "approved": critique["approved"]
        }
