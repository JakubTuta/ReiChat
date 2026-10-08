"""App-wide settings the user changes at runtime (stored in AppSetting),
as opposed to django_server/settings.py, which is fixed at startup."""

from . import models

EXTERNAL_OLLAMA_URL = "external_ollama_url"


def get(key: str, default: object = None) -> object:
    setting = models.AppSetting.objects.filter(key=key).first()
    return setting.value if setting is not None else default


def put(key: str, value: object) -> None:
    models.AppSetting.objects.update_or_create(key=key, defaults={"value": value})


def external_ollama_url() -> str | None:
    """Set when the user runs their own Ollama instead of letting the app
    manage one container per model — every model then lives there."""
    value = get(EXTERNAL_OLLAMA_URL)
    return value.rstrip("/") if isinstance(value, str) and value else None


def set_external_ollama_url(url: str | None) -> None:
    put(EXTERNAL_OLLAMA_URL, url or "")
