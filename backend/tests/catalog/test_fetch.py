import httpx
import pytest
import respx

from app.catalog import check_live
from app.catalog.parser import LIBRARY_URL, fetch_library_html
from tests.catalog.test_parser import FIXTURE_PATH


@respx.mock
async def test_fetch_library_html_returns_the_page_and_identifies_itself():
    route = respx.get(LIBRARY_URL).respond(200, text="<html>hi</html>")

    assert await fetch_library_html() == "<html>hi</html>"
    assert "chatbot-catalog-sync" in route.calls.last.request.headers["user-agent"]


@respx.mock
async def test_fetch_library_html_follows_redirects():
    respx.get(LIBRARY_URL).respond(301, headers={"Location": "https://ollama.com/library/"})
    respx.get("https://ollama.com/library/").respond(200, text="moved")

    assert await fetch_library_html() == "moved"


@respx.mock
async def test_fetch_library_html_raises_on_an_error_status():
    respx.get(LIBRARY_URL).respond(503)

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_library_html()


@respx.mock
async def test_check_live_passes_when_the_page_parses_into_enough_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(check_live, "MIN_EXPECTED_MODELS", 4)
    respx.get(LIBRARY_URL).respond(200, text=FIXTURE_PATH.read_text(encoding="utf-8"))

    exit_code = await check_live.main()

    assert exit_code == 0
    assert "Parsed 4 models" in capsys.readouterr().out


@respx.mock
async def test_check_live_fails_when_far_fewer_models_than_expected_parse(capsys: pytest.CaptureFixture[str]):
    respx.get(LIBRARY_URL).respond(200, text=FIXTURE_PATH.read_text(encoding="utf-8"))

    exit_code = await check_live.main()

    assert exit_code == 1
    assert "Only 4 models parsed" in capsys.readouterr().out


@respx.mock
async def test_check_live_fails_when_the_markup_is_unrecognisable(capsys: pytest.CaptureFixture[str]):
    respx.get(LIBRARY_URL).respond(200, text="<html><body>redesigned</body></html>")

    exit_code = await check_live.main()

    assert exit_code == 1
    assert "::error::" in capsys.readouterr().out


@respx.mock
async def test_check_live_fails_when_ollama_com_is_unreachable(capsys: pytest.CaptureFixture[str]):
    respx.get(LIBRARY_URL).mock(side_effect=httpx.ConnectError("offline"))

    exit_code = await check_live.main()

    assert exit_code == 1
    assert "Could not reach ollama.com" in capsys.readouterr().out
