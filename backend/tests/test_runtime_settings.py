import pytest
from sqlalchemy import func, select

from app import runtime_settings
from app.db.models import AppSetting
from app.db.session import SessionLocal

# conftest replaces the module's attribute for every test; keep the real one.
REAL_EXTERNAL_OLLAMA_URL = runtime_settings.external_ollama_url

pytestmark = pytest.mark.usefixtures("db")


async def test_an_unset_key_gives_the_default():
    assert await runtime_settings.get("missing") is None
    assert await runtime_settings.get("missing", "fallback") == "fallback"


async def test_a_value_survives_a_round_trip():
    await runtime_settings.put("gpu_share", {"percent": 50})

    assert await runtime_settings.get("gpu_share") == {"percent": 50}


async def test_putting_the_same_key_again_replaces_the_value():
    await runtime_settings.put("k", "first")
    await runtime_settings.put("k", "second")

    async with SessionLocal() as session:
        rows = (await session.execute(select(func.count()).select_from(AppSetting))).scalar_one()
    assert rows == 1
    assert await runtime_settings.get("k") == "second"


async def test_a_value_can_be_cleared_to_none():
    await runtime_settings.put("k", "value")
    await runtime_settings.put("k", None)

    assert await runtime_settings.get("k", "default") is None


async def test_no_external_ollama_means_app_managed_containers():
    assert await REAL_EXTERNAL_OLLAMA_URL() is None


async def test_the_external_ollama_url_is_returned_without_a_trailing_slash():
    await runtime_settings.set_external_ollama_url("http://host.docker.internal:11434/")

    assert await REAL_EXTERNAL_OLLAMA_URL() == "http://host.docker.internal:11434"


async def test_clearing_the_external_ollama_url_goes_back_to_app_managed():
    await runtime_settings.set_external_ollama_url("http://host.docker.internal:11434")
    await runtime_settings.set_external_ollama_url(None)

    assert await REAL_EXTERNAL_OLLAMA_URL() is None
    assert await runtime_settings.get(runtime_settings.EXTERNAL_OLLAMA_URL) == ""


@pytest.mark.parametrize("junk", [123, ["http://x"], True, {"url": "x"}])
async def test_a_stored_value_that_is_not_a_url_string_counts_as_not_set(junk):
    await runtime_settings.put(runtime_settings.EXTERNAL_OLLAMA_URL, junk)

    assert await REAL_EXTERNAL_OLLAMA_URL() is None
