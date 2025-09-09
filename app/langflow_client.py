from __future__ import annotations
import os
import requests
from typing import List, Dict, Any

class LangflowClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, flow_id: str | None = None):
        self.base_url = (base_url or os.getenv("LANGFLOW_BASE_URL", "http://localhost:7860")).rstrip("/")
        self.api_key = api_key or os.getenv("LANGFLOW_API_KEY", "")
        self.flow_id = flow_id or os.getenv("LANGFLOW_FLOW_ID", "multi-agent-rag")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})

    def run(self, message: str, history: List[Dict[str, str]] | None = None, extras: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/run/{self.flow_id}"
        payload = {
            "input_value": message,
            "input_type": "chat",
            "output_type": "chat",
            "stream": False,
            "history": history or [],
            "extras": extras or {},
        }
        resp = self.session.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/api/v1/health", timeout=10)
            return r.ok
        except Exception:
            return False
