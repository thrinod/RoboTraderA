from datetime import datetime

class AgentLogger:
    def __init__(self):
        self.logs = []

    def log(self, agent_name: str, action: str, message: str, data: dict = None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "action": action,
            "message": message,
            "data": data or {}
        }
        self.logs.append(log_entry)
        # Keep only the last 100 logs to avoid memory bloat
        if len(self.logs) > 100:
            self.logs.pop(0)

    def get_logs(self):
        return self.logs

agent_logger = AgentLogger()
