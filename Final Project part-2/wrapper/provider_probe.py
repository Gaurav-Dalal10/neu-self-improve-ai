import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def probe(base_url: str, api_key: str, model: str, timeout_sec: int = 30) -> dict:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    t0 = time.time()
    try:
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_sec) as r:
            txt = r.read().decode("utf-8")
        return {"ok": True, "latency_sec": round(time.time() - t0, 3), "response": txt[:1000]}
    except urllib.error.HTTPError as e:
        return {"ok": False, "latency_sec": round(time.time() - t0, 3), "error": f"HTTPError {e.code}: {e.reason}"}
    except Exception as e:
        return {"ok": False, "latency_sec": round(time.time() - t0, 3), "error": str(e)}


def main() -> None:
    p = argparse.ArgumentParser(description="Probe OpenAI-compatible provider endpoint/model")
    p.add_argument("--base_url", required=True)
    p.add_argument("--api_key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--log", default="final_project_part2/outputs/qwen35_sweep/probe.log")
    args = p.parse_args()

    res = probe(args.base_url, args.api_key, args.model)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"model": args.model, **res}) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
