import pytest

from app import runtime_settings

# conftest replaces the module's attribute for every test; keep the real one.
REAL_EXTERNAL_OLLAMA_URL = runtime_settings.external_ollama_url

pytestmark = pytest.mark.usefixtures("db")


async def test_an_unset_key_gives_the_default():
    assert await runtime_settings.get("missing") is None
    assert await runtime_settings.get("missing", "fallback") == "fallback"


async def test_a_value_survives_a_round_trip():
    await runtime_settings.put("gpu_share", {"percent": 50})

    assert await runtime_settings.get("gpu_share") == {"percent": 50}


async def test_no_external_ollama_means_app_managed_containers():
    assert await REAL_EXTERNAL_OLLAMA_URL() is None


async def test_the_external_ollama_url_is_returned_without_a_trailing_slash():
    await runtime_settings.set_external_ollama_url("http://host.docker.internal:11434/")

    assert await REAL_EXTERNAL_OLLAMA_URL() == "http://host.docker.internal:11434"


@pytest.mark.parametrize("junk", [123, ["http://x"]])
async def test_a_stored_value_that_is_not_a_url_string_counts_as_not_set(junk):
    await runtime_settings.put(runtime_settings.EXTERNAL_OLLAMA_URL, junk)

    assert await REAL_EXTERNAL_OLLAMA_URL() is None
