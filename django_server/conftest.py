import pytest
from django.core.cache import cache


@pytest.fixture(autouse=True)
def _reset_request_throttle():
    # DRF's anon throttle counts requests in the default cache, which lives
    # for the whole test run — past 120 requests a minute, unrelated tests
    # start getting 429s depending on how fast the suite runs.
    cache.clear()
    yield


@pytest.fixture(autouse=True)
def _app_managed_runtime(monkeypatch):
    # Reading the runtime setting is a DB query, which most tests here can't
    # make. Default every test to app-managed containers; tests about the
    # user's own Ollama override this.
    from django_app import app_settings

    monkeypatch.setattr(app_settings, "external_ollama_url", lambda: None)

