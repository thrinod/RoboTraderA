from core.llm_provider import ollama_client
from core.logger import agent_logger

class AnalysisAgent:
    def __init__(self):
        self.system_prompt = (
            "You are an expert technical and fundamental analysis agent. "
            "Your task is to analyze trading data and provide a concise summary of the market conditions."
        )

    def analyze(self, symbol: str, context: str) -> str:
        prompt = f"Please analyze the following asset: {symbol}. Context: {context}"
        
        agent_logger.log("AnalysisAgent", "Started Analysis", f"Analyzing {symbol}", {"context": context})
        
        try:
            analysis_result = ollama_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt
            )
            agent_logger.log("AnalysisAgent", "Completed Analysis", f"Finished analysis for {symbol}", {"result": analysis_result})
            return analysis_result
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            agent_logger.log("AnalysisAgent", "Error", error_msg)
            return error_msg
