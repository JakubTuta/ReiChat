"""Parses the ollama.com library index page into structured model data.

The index page (https://ollama.com/library) lists every model with its
description, capability chips (vision/tools/thinking/embedding/audio) and
size chips (1b/8b/70b/...) inline in a single response — no per-model page
fetch is needed. This replaces the old scraper, which fetched one page per
model (231+ requests with a 1s sleep each) and gated every result on an
`x-test-pull-count` attribute ollama.com has since removed.

Markup verified live against https://ollama.com/library:

    <li class="flex items-baseline ...">
      <a href="/library/llama3.1">
        <p class="max-w-lg break-words ...">Llama 3.1 is ...</p>
        <span class="... bg-indigo-50 ...">tools</span>          <!-- capability -->
        <span class="... bg-[#ddf4ff] ...">8b</span>             <!-- variant -->
        ...
        <span>118.3M</span> ... <span class="hidden sm:flex">Pulls</span>
      </a>
    </li>

If ollama.com changes this markup again, `parse_library` raises
CatalogParseError instead of silently returning an empty list — the failure
mode that caused the original bug (0 models parsed, reported as success).
"""

from __future__ import annotations

import dataclasses
import logging

import httpx
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

LIBRARY_URL = "https://ollama.com/library"
_TIMEOUT_SECONDS = 30.0
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) chatbot-catalog-sync/1.0"
)

_CAPABILITY_CHIP_MARKER = "bg-indigo-50"
_VARIANT_CHIP_MARKER = "bg-[#ddf4ff]"
KNOWN_CAPABILITIES = {"vision", "tools", "thinking", "embedding", "audio"}
DEFAULT_VARIANT = "latest"


class CatalogParseError(Exception):
    """Raised when the library page cannot be parsed into any models.

    Never swallow this into an empty list — the caller must treat it as a
    failed refresh, not a catalog of zero models.
    """


@dataclasses.dataclass(frozen=True)
class ParsedModel:
    name: str
    description: str
    popularity: int
    capabilities: frozenset[str]
    variants: tuple[str, ...]

    @property
    def can_process_image(self) -> bool:
        return "vision" in self.capabilities

    @property
    def can_use_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def is_embedding(self) -> bool:
        return self.capabilities == frozenset({"embedding"})


async def fetch_library_html() -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=_TIMEOUT_SECONDS) as client:
        response = await client.get(LIBRARY_URL, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
    return response.text


def parse_pull_count(text: str) -> int:
    """ "118.3M" -> 118300000, "1,234" -> 1234, "" -> 0."""
    text = text.strip().upper().replace(",", "")
    if not text:
        return 0

    multiplier = 1.0
    for suffix, factor in (("K", 1_000.0), ("M", 1_000_000.0), ("B", 1_000_000_000.0)):
        if text.endswith(suffix):
            multiplier = factor
            text = text[: -len(suffix)]
            break

    try:
        return int(float(text) * multiplier)
    except ValueError:
        logger.warning("Could not parse pull count %r", text)
        return 0


def _class_text(tag: Tag) -> str:
    # bs4 hands `class` back as a list of names, but a plain string when the
    # attribute isn't multi-valued.
    classes = tag.get("class")
    if classes is None:
        return ""
    return classes if isinstance(classes, str) else " ".join(classes)


def _chip_texts(item: Tag, class_marker: str) -> list[str]:
    return [span.get_text(strip=True) for span in item.find_all("span") if class_marker in _class_text(span)]


def _parse_pulls(item: Tag) -> int:
    # The value span has no distinguishing class; the label span next to it
    # does ("Pulls"). Anchor on the label and read its previous sibling so
    # this survives the metrics being reordered.
    label = item.find("span", string=lambda s: bool(s) and s.strip() == "Pulls")
    if label is None:
        return 0
    value_span = label.find_previous_sibling("span")
    if value_span is None:
        return 0
    return parse_pull_count(value_span.get_text())


def _parse_item(item: Tag) -> ParsedModel | None:
    link = item.find("a", href=True)
    if not isinstance(link, Tag):
        return None
    href = link.get("href")
    if not isinstance(href, str) or not href.startswith("/library/"):
        return None

    name = href.removeprefix("/library/").strip()
    if not name:
        return None

    description_tag = item.find("p", class_="max-w-lg")
    description = description_tag.get_text(strip=True) if description_tag else ""

    capabilities = frozenset(_chip_texts(item, _CAPABILITY_CHIP_MARKER)) & KNOWN_CAPABILITIES
    variants = tuple(_chip_texts(item, _VARIANT_CHIP_MARKER)) or (DEFAULT_VARIANT,)
    popularity = _parse_pulls(item)

    return ParsedModel(
        name=name,
        description=description,
        popularity=popularity,
        capabilities=capabilities,
        variants=variants,
    )


def parse_library(html: str) -> list[ParsedModel]:
    soup = BeautifulSoup(html, "html.parser")

    items = [item for item in soup.find_all("li") if "items-baseline" in _class_text(item)]

    models: list[ParsedModel] = []
    for item in items:
        parsed = _parse_item(item)
        if parsed is not None:
            models.append(parsed)

    if not models:
        raise CatalogParseError(
            "Parsed 0 models from the ollama.com library page — the site's "
            "markup has likely changed (expected <li> items with class "
            "'items-baseline' containing a /library/<name> link)."
        )

    return models
