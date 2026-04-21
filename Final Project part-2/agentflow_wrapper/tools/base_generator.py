from typing import Dict, List


def base_generate(client, messages: List[Dict[str, str]], max_tokens: int = 512) -> Dict[str, object]:
    resp = client.chat_completion(messages=messages, max_tokens=max_tokens)
    return {
        "status": "ok",
        "content": resp.content,
        "reasoning_content": resp.reasoning_content,
        "raw": resp.raw,
    }
