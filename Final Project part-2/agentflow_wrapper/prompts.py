import json
from typing import Any, Dict, List


def planner_prompt(question: str, memory: List[Dict[str, Any]], tools: List[str]) -> List[Dict[str, str]]:
    schema = {
        "tool": "one of available tools",
        "subgoal": "short subgoal",
        "command": "tool argument string",
    }
    sys = (
        "You are Planner. Output valid JSON only with keys: tool, subgoal, command. "
        "Choose exactly one tool."
    )
    user = {
        "question": question,
        "memory": memory[-5:],
        "available_tools": tools,
        "required_json_schema": schema,
    }
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
    ]


def verifier_prompt(question: str, memory: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sys = "You are Verifier. Reply exactly as JSON: {\"decision\":\"PASS|FAIL\",\"reason\":\"...\"}."
    user = {"question": question, "memory": memory[-8:]}
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
    ]


def generator_prompt(question: str, memory: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sys = "You are Generator. Produce concise final answer from memory evidence only."
    user = {"question": question, "memory": memory[-12:]}
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
    ]
