from core.logger import agent_logger
import json
import os

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "rl_memory.json")

class RLAgent:
    def __init__(self):
        self.memory = self._load_memory()
        
    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_memory(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def get_learned_context(self, symbol: str) -> str:
        """Returns insights based on past successful/failed trades for this symbol."""
        symbol_mem = self.memory.get(symbol, {"successes": 0, "failures": 0, "lessons": []})
        
        context = (
            f"RL History for {symbol}: {symbol_mem['successes']} successful trades, "
            f"{symbol_mem['failures']} failed trades. "
            f"Lessons learned: {', '.join(symbol_mem['lessons']) if symbol_mem['lessons'] else 'None yet.'}"
        )
        
        agent_logger.log("RLAgent", "Providing Context", f"Loaded past training data for {symbol}", {"memory_state": symbol_mem})
        return context

    def update_policy(self, symbol: str, action: str, outcome_reward: float, lesson: str = ""):
        """Called after a trade is resolved to update the reinforcement memory."""
        if symbol not in self.memory:
            self.memory[symbol] = {"successes": 0, "failures": 0, "lessons": []}
            
        if outcome_reward > 0:
            self.memory[symbol]["successes"] += 1
        else:
            self.memory[symbol]["failures"] += 1
            
        if lesson and lesson not in self.memory[symbol]["lessons"]:
            self.memory[symbol]["lessons"].append(lesson)
            # keep only last 5 lessons
            if len(self.memory[symbol]["lessons"]) > 5:
                self.memory[symbol]["lessons"].pop(0)
                
        self._save_memory()
        agent_logger.log("RLAgent", "Training Update", f"Updated policy weights/memory for {symbol}", {"reward": outcome_reward, "lesson": lesson})
