import json
from typing import Any, Dict, List

from ..prompts import verifier_prompt


class Verifier:
    def __init__(self, client):
        self.client = client

    def verify(self, question: str, memory: List[Dict[str, Any]]) -> Dict[str, str]:
        messages = verifier_prompt(question, memory)
        resp = self.client.chat_completion(messages=messages, temperature=0.0, max_tokens=120)
        try:
            obj = json.loads(resp.content)
            decision = obj.get("decision", "FAIL")
            reason = obj.get("reason", "")
        except Exception:
            decision = "FAIL"
            reason = "Unparseable verifier output"
        if decision not in {"PASS", "FAIL"}:
            decision = "FAIL"
        return {"decision": decision, "reason": str(reason)}
