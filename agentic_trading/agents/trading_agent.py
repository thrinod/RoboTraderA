from core.llm_provider import ollama_client
from core.logger import agent_logger

class TradingAgent:
    def __init__(self):
        self.system_prompt = (
            "You are an expert trading execution agent. "
            "Based on the analysis provided, recommend a trading action: BUY, SELL, or HOLD. "
            "Provide a brief justification."
        )

    def decide(self, symbol: str, analysis: str) -> str:
        prompt = f"Asset: {symbol}\nMarket Analysis: {analysis}\nWhat is your recommended action?"
        
        agent_logger.log("TradingAgent", "Evaluating Decision", f"Evaluating action for {symbol}", {"analysis": analysis})
        try:
            decision = ollama_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt
            )
            agent_logger.log("TradingAgent", "Decision Made", f"Action for {symbol}: {decision}")
            return decision
        except Exception as e:
            error_msg = f"Error during decision making: {str(e)}"
            agent_logger.log("TradingAgent", "Error", error_msg)
            return error_msg
