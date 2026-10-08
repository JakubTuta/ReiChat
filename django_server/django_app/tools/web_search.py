"""The opt-in web search tool. Unlike the built-ins it reaches the network,
so it's only offered while the user has turned web search on (see
container/searxng.py), and every search shows up in the reply's tool trace.
"""

import requests
from langchain_core.tools import tool

from container import searxng

MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 400
REQUEST_TIMEOUT_SECONDS = 15


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use it for recent events,
    facts you're unsure of, or anything after your training data. Returns
    the top results with title, link and a short excerpt.
    """
    base = searxng.base_url()
    if base is None:
        return "Error: web search is turned off or not running."

    try:
        response = requests.get(
            f"{base}/search", params={"q": query, "format": "json"}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    except (requests.exceptions.RequestException, ValueError) as e:
        return f"Error: the web search failed ({e})."

    if not results:
        return f"No web results for {query!r}."

    lines = []
    for result in results[:MAX_RESULTS]:
        snippet = (result.get("content") or "").strip()[:MAX_SNIPPET_CHARS]
        lines.append(f"{result.get('title', '').strip()}\n{result.get('url', '')}\n{snippet}")
    return "\n\n".join(lines)


def available() -> bool:
    return searxng.status() == "running"
