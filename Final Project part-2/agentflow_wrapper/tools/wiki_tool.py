import json
import os
from typing import Dict, List


def wikipedia_search(query: str, wiki_jsonl_path: str = "") -> Dict[str, object]:
    if not wiki_jsonl_path or not os.path.exists(wiki_jsonl_path):
        return {
            "query": query,
            "results": [],
            "status": "missing_wiki_source",
            "note": "Provide local wiki JSONL path.",
        }

    results: List[Dict[str, str]] = []
    with open(wiki_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            title = row.get("title", "")
            text = row.get("text", "")
            if query.lower() in title.lower() or query.lower() in text.lower():
                results.append({"title": title, "text": text[:500]})
            if len(results) >= 3:
                break

    return {"query": query, "results": results, "status": "ok" if results else "no_results"}
