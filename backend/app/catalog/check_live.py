"""Standalone script: parse the live ollama.com library page and fail loudly
if it comes back empty or suspiciously small. Run by the weekly
catalog-drift-check CI workflow (and by hand: `python -m app.catalog.check_live`)
so an ollama.com redesign is caught here instead of by a user opening
`/models` to an empty page — the exact failure mode that started this
overhaul (see catalog/parser.py's module docstring).

Deliberately dependency-light: only needs httpx + beautifulsoup4, not the
full app, so CI doesn't need a database or the rest of the dependencies.
"""

from __future__ import annotations

import asyncio
import sys

from .parser import CatalogParseError, fetch_library_html, parse_library

MIN_EXPECTED_MODELS = 50  # verified live at ~231; well under that is a real problem, not noise


async def main() -> int:
    try:
        html = await fetch_library_html()
        models = parse_library(html)
    except CatalogParseError as e:
        print(f"::error::{e}")
        return 1
    except Exception as e:
        print(f"::error::Could not reach ollama.com: {e}")
        return 1

    print(f"Parsed {len(models)} models")

    if len(models) < MIN_EXPECTED_MODELS:
        print(
            f"::error::Only {len(models)} models parsed — expected {MIN_EXPECTED_MODELS}+. "
            "ollama.com's markup has likely changed again."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
