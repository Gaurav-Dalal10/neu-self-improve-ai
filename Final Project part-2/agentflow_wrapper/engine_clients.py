import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class EngineClientError(RuntimeError):
    pass


def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    redacted = dict(headers)
    if "Authorization" in redacted:
        redacted["Authorization"] = "Bearer ***REDACTED***"
    return redacted


@dataclass
class ChatResponse:
    content: str
    raw: Dict[str, Any]
    reasoning_content: Optional[str] = None


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: int = 60,
        max_retries: int = 3,
        log_dir: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.log_dir = log_dir

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        return payload

    def _log_io(self, payload: Dict[str, Any], headers: Dict[str, str], response: Dict[str, Any]) -> None:
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.join(self.log_dir, f"request_{ts}.json")
        obj = {
            "endpoint": self._endpoint(),
            "headers": _redact_headers(headers),
            "payload": payload,
            "response": response,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        request_timeout_sec: Optional[int] = None,
    ) -> ChatResponse:
        payload = self._build_payload(messages, temperature, top_p, max_tokens, stop)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        timeout = request_timeout_sec or self.timeout_sec
        err: Optional[Exception] = None

        for i in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(self._endpoint(), data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw_text = resp.read().decode("utf-8")
                data = json.loads(raw_text)
                self._log_io(payload, headers, data)

                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("content", "")
                reasoning = msg.get("reasoning_content")
                return ChatResponse(content=content, raw=data, reasoning_content=reasoning)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                err = e
                if i >= self.max_retries:
                    break
                time.sleep(min(2 ** i, 8))

        raise EngineClientError(f"Request failed after retries: {err}")


class DashScopeOpenAICompatibleClient(OpenAICompatibleClient):
    pass


def make_client(
    engine_type: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout_sec: int = 60,
    max_retries: int = 3,
    log_dir: Optional[str] = None,
):
    if engine_type == "openai_compat":
        return OpenAICompatibleClient(base_url, api_key, model, timeout_sec, max_retries, log_dir)
    if engine_type == "dashscope_compat":
        return DashScopeOpenAICompatibleClient(base_url, api_key, model, timeout_sec, max_retries, log_dir)
    raise ValueError(f"Unsupported engine type: {engine_type}")
