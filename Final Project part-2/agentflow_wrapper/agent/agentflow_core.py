import json
import os
from typing import Any, Dict, List, Optional

from ..prompts import generator_prompt
from ..tools.base_generator import base_generate
from .executor import Executor
from .memory import AgentMemory
from .planner import Planner
from .verifier import Verifier


class AgentFlowCore:
    def __init__(self, client, wiki_jsonl_path: str = ""):
        self.client = client
        self.tools = ["base_generator", "python_coder", "wikipedia_search", "web_search", "google_search"]
        self.planner = Planner(client, self.tools)
        self.executor = Executor(client, wiki_jsonl_path=wiki_jsonl_path)
        self.verifier = Verifier(client)

    def run(self, question: str, max_turns: int = 4) -> Dict[str, Any]:
        memory = AgentMemory()
        trace: List[Dict[str, Any]] = []

        for t in range(max_turns):
            action = self.planner.plan(question, memory.as_list())
            exec_result = self.executor.execute(action)
            step = {
                "turn": t + 1,
                "action": action,
                "execution": exec_result,
            }
            memory.add(step)
            trace.append(step)

            verdict = self.verifier.verify(question, memory.as_list())
            step["verifier"] = verdict
            if verdict["decision"] == "PASS":
                break

        gen_messages = generator_prompt(question, memory.as_list())
        final = base_generate(self.client, gen_messages, max_tokens=300)

        return {
            "question": question,
            "trace": trace,
            "final_answer": final.get("content", ""),
            "status": "ok",
        }

    @staticmethod
    def save_trace(obj: Dict[str, Any], out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
