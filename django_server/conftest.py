import pytest
from django.core.cache import cache


@pytest.fixture(autouse=True)
def _reset_request_throttle():
    # DRF's anon throttle counts requests in the default cache, which lives
    # for the whole test run — past 120 requests a minute, unrelated tests
    # start getting 429s depending on how fast the suite runs.
    cache.clear()
    yield
