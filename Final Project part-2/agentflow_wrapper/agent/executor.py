from typing import Any, Dict

from ..tools.base_generator import base_generate
from ..tools.python_tool import run_python
from ..tools.search_tools import google_search, web_search
from ..tools.wiki_tool import wikipedia_search


class Executor:
    def __init__(self, client, wiki_jsonl_path: str = ""):
        self.client = client
        self.wiki_jsonl_path = wiki_jsonl_path

    def execute(self, action: Dict[str, str]) -> Dict[str, Any]:
        tool = action.get("tool", "base_generator")
        command = action.get("command", "")

        if tool == "python_coder":
            return {"tool": tool, "output": run_python(command)}
        if tool == "wikipedia_search":
            return {"tool": tool, "output": wikipedia_search(command, self.wiki_jsonl_path)}
        if tool == "google_search":
            return {"tool": tool, "output": google_search(command)}
        if tool == "web_search":
            return {"tool": tool, "output": web_search(command)}

        msg = [{"role": "user", "content": command or "Answer the question."}]
        return {"tool": "base_generator", "output": base_generate(self.client, msg)}
