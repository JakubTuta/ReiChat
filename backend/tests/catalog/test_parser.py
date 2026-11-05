"""Regression guard for the reported bug: the old scraper gated every model
on an `x-test-pull-count` attribute ollama.com removed, so it silently
parsed 0 models and reported success. `parse_library` must instead raise
`CatalogParseError` when it finds nothing, and must correctly extract every
field from real ollama.com markup.

`library_index_sample.html` is a trimmed, real fixture (4 <li> items cut
from a live fetch of https://ollama.com/library) — not synthetic HTML — so
this test only passes if the parser handles the actual site structure.
"""

import pathlib
import unittest

from app.catalog.parser import CatalogParseError, parse_library, parse_pull_count

FIXTURE_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / "library_index_sample.html"


class ParsePullCountTests(unittest.TestCase):
    def test_millions(self):
        self.assertEqual(parse_pull_count("118.3M"), 118_300_000)

    def test_thousands(self):
        self.assertEqual(parse_pull_count("5K"), 5_000)


class ParseLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = FIXTURE_PATH.read_text(encoding="utf-8")
        cls.models = {m.name: m for m in parse_library(cls.html)}

    def test_parses_every_item(self):
        self.assertEqual(set(self.models), {"llama3.1", "deepseek-r1", "nomic-embed-text", "gemma3"})

    def test_variant_chips_and_popularity(self):
        llama = self.models["llama3.1"]
        self.assertEqual(llama.variants, ("8b", "70b", "405b"))
        self.assertEqual(llama.popularity, 118_300_000)
        self.assertFalse(llama.can_process_image)
        self.assertFalse(llama.is_embedding)

    def test_embedding_only_model_flagged_and_defaults_to_latest(self):
        # nomic-embed-text has no size chips on ollama.com — must default
        # to the "latest" tag rather than being dropped.
        embed = self.models["nomic-embed-text"]
        self.assertEqual(embed.capabilities, frozenset({"embedding"}))
        self.assertTrue(embed.is_embedding)
        self.assertEqual(embed.variants, ("latest",))

    def test_tools_capability_drives_can_use_tools(self):
        deepseek = self.models["deepseek-r1"]
        self.assertTrue(deepseek.can_use_tools)
        embed = self.models["nomic-embed-text"]
        self.assertFalse(embed.can_use_tools)

    def test_empty_page_fails_loudly_instead_of_reporting_zero_models(self):
        with self.assertRaises(CatalogParseError):
            parse_library("<html><body><p>ollama.com redesigned this page</p></body></html>")


if __name__ == "__main__":
    unittest.main()
