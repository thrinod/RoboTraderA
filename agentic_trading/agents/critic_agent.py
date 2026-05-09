from core.llm_provider import ollama_client
from core.logger import agent_logger

class CriticAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a strict, risk-averse Critic Agent. Your job is to review a proposed trading decision. "
            "You will look at historical performance, potential downside, and the analysis provided. "
            "You must output either 'APPROVED' or 'REJECTED' followed by a short explanation."
        )

    def critique(self, symbol: str, analysis: str, proposed_decision: str, historical_context: str) -> dict:
        prompt = (
            f"Asset: {symbol}\n"
            f"Analysis: {analysis}\n"
            f"Proposed Action: {proposed_decision}\n"
            f"Historical Context: {historical_context}\n"
            "Evaluate this decision. Start your response with APPROVED or REJECTED."
        )
        
        agent_logger.log("CriticAgent", "Reviewing Decision", f"Critiquing proposed {proposed_decision} for {symbol}", {"historical_context": historical_context})
        
        try:
            critique_result = ollama_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt
            )
            
            is_approved = "APPROVED" in critique_result.upper()
            
            agent_logger.log("CriticAgent", "Critique Complete", f"Result: {'APPROVED' if is_approved else 'REJECTED'}", {"details": critique_result})
            
            return {
                "approved": is_approved,
                "feedback": critique_result
            }
        except Exception as e:
            error_msg = f"Error during critique: {str(e)}"
            agent_logger.log("CriticAgent", "Error", error_msg)
            return {"approved": False, "feedback": error_msg}
