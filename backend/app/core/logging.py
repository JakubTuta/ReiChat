import logging.config
from typing import Any

LOGGING: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    # No handler of its own: records propagate to the root console handler, so
    # pytest's caplog (attached to root) sees them too.
    "loggers": {
        "app": {"level": "DEBUG"},
    },
}


def configure_logging() -> None:
    logging.config.dictConfig(LOGGING)
