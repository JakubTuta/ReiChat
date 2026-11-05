import pytest

from app.core.config import Settings


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_debug_is_on_only_for_the_word_true(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool):
    monkeypatch.setenv("DEBUG", value)

    assert Settings().debug is expected


@pytest.mark.parametrize("value", ["*", "express:*"])
def test_a_stray_debug_value_from_another_tool_neither_crashes_nor_enables_it(
    monkeypatch: pytest.MonkeyPatch, value: str
):
    monkeypatch.setenv("DEBUG", value)

    assert Settings().debug is False


def test_debug_defaults_to_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEBUG", raising=False)

    assert Settings().debug is False
