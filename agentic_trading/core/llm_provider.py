import requests
import json
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma4:e2b")

class OllamaProvider:
    def __init__(self, base_url=OLLAMA_BASE_URL, default_model=DEFAULT_MODEL):
        self.base_url = base_url
        self.default_model = default_model

    def generate(self, prompt: str, model: str = None, system_prompt: str = None):
        model_to_use = model or self.default_model
        
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Error from Ollama: {response.text}")

ollama_client = OllamaProvider()
