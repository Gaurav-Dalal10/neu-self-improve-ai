import json
from typing import Any, Dict, List

from ..prompts import planner_prompt


def parse_tool_choice(text: str, tools: List[str]) -> Dict[str, str]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = {"tool": "base_generator", "subgoal": "fallback", "command": ""}

    tool = obj.get("tool", "base_generator")
    if tool not in tools:
        tool = "base_generator"
    return {
        "tool": tool,
        "subgoal": str(obj.get("subgoal", "")),
        "command": str(obj.get("command", "")),
    }


class Planner:
    def __init__(self, client, tools: List[str]):
        self.client = client
        self.tools = tools

    def plan(self, question: str, memory: List[Dict[str, Any]]) -> Dict[str, str]:
        messages = planner_prompt(question, memory, self.tools)
        resp = self.client.chat_completion(messages=messages, temperature=0.0, max_tokens=200)
        return parse_tool_choice(resp.content, self.tools)
