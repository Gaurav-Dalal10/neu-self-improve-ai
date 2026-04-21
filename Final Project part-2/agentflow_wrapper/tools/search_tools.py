from typing import Dict, List


def web_search(query: str) -> Dict[str, object]:
    # Offline-safe placeholder tool.
    return {
        "query": query,
        "results": [],
        "status": "no_results",
        "note": "Offline mode: no external web search configured.",
    }


def google_search(query: str) -> Dict[str, object]:
    return web_search(query)
