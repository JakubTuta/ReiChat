"""The opt-in web search tool. Unlike the built-ins it reaches the network,
so it's only offered while the user has turned web search on (see
containers/searxng.py), and every search shows up in the reply's tool trace.
"""

import httpx
from langchain_core.tools import tool

from app.containers import searxng

MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 400
REQUEST_TIMEOUT_SECONDS = 15.0


@tool
async def web_search(query: str) -> str:
    """Search the web for current information. Use it for recent events,
    facts you're unsure of, or anything after your training data. Returns
    the top results with title, link and a short excerpt.
    """
    base = await searxng.base_url()
    if base is None:
        return "Error: web search is turned off or not running."

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{base}/search", params={"q": query, "format": "json"})
        response.raise_for_status()
        results = response.json().get("results") or []
    except (httpx.HTTPError, ValueError) as e:
        return f"Error: the web search failed ({e})."

    if not results:
        return f"No web results for {query!r}."

    lines = []
    for result in results[:MAX_RESULTS]:
        snippet = (result.get("content") or "").strip()[:MAX_SNIPPET_CHARS]
        lines.append(f"{result.get('title', '').strip()}\n{result.get('url', '')}\n{snippet}")
    return "\n\n".join(lines)


async def available() -> bool:
    return await searxng.status() == "running"
