from typing import Any, Dict, List

from ..prompts import generator_prompt
from ..tools.base_generator import base_generate


class Generator:
    def __init__(self, client):
        self.client = client

    def generate(self, question: str, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages = generator_prompt(question, memory)
        return base_generate(self.client, messages, max_tokens=300)
