import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# This app is single-user, unauthenticated and bound to 127.0.0.1 (see
# docker-compose.yaml). There is no deployment mode: nothing here is meant to
# be reachable from another machine, so these values are constants rather than
# knobs. The two exceptions below (DOCKER, DEBUG) are the only things that
# genuinely differ between running in compose and running from source.

# Set by the Dockerfile. Inside the compose network postgres and redis answer
# to their service names; from source they are the loopback ports compose
# publishes. The local address is an IP literal, not "localhost" — Windows
# resolves "localhost" to ::1 first and Docker only publishes on IPv4, so a
# hostname hangs for a long OS-level timeout before falling back.
IS_DOCKER = os.getenv("DOCKER", "false") == "true"

# No auth, no sessions, no logins — the only thing this still signs is the CSRF
# token, protecting a stack only the local user can reach.
SECRET_KEY = "django-insecure-local-single-user-key"

# Renders stack traces and settings on the error page, so it stays off unless
# you deliberately ask for it while running from source.
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Not "*": asgi.py wraps the WebSocket router in AllowedHostsOriginValidator,
# so a permissive value here would also accept a WebSocket handshake with *any*
# Origin, meaning any web page the user has open could drive their local chat
# sockets. Binding ports to 127.0.0.1 (docker-compose.yaml) stops other
# machines on the network, but it does nothing against a malicious page running
# in the user's own browser, which can always reach localhost — this is the
# actual defense for that case.
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# The frontend, on the port compose publishes for it, under both names the
# user may type. A browser treats them as different origins: missing one here
# makes every response to that tab unreadable (no Access-Control-Allow-Origin),
# so the app reports the backend as unreachable while Django logs 200s.
CORS_ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    "UNAUTHENTICATED_USER": None,
    # Every request is anonymous (no auth exists), so this is the only
    # throttle that applies — a floor against a malicious page firing
    # blind requests (no CORS read access needed to send them) or a
    # runaway client loop, not a real per-user rate limit.
    "DEFAULT_THROTTLE_CLASSES": ["rest_framework.throttling.AnonRateThrottle"],
    "DEFAULT_THROTTLE_RATES": {"anon": "120/min"},
    # JSON only — form/multipart bodies are what a cross-origin page can send
    # without a CORS preflight. The one upload endpoint (Documents) opts back
    # into multipart itself; see helpers/middleware.py for the actual guard.
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
}

INSTALLED_APPS = [
    "corsheaders",
    "rest_framework",
    "channels",
    "daphne",
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "django_app",
    "container",
]

ASGI_APPLICATION = "django_server.asgi.application"

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("redis" if IS_DOCKER else "127.0.0.1", 6379)],
        },
    }
}

MIDDLEWARE = [
    # Must precede CommonMiddleware (and anything else that can generate a
    # response) or CORS headers silently don't get attached to those
    # responses.
    "corsheaders.middleware.CorsMiddleware",
    "helpers.middleware.OriginGuardMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "django_server.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "django_server.wsgi.application"


# Credentials are the ones docker-compose.yaml gives the postgres service —
# both sides have to agree, so changing one means changing the other. The
# database only ever listens on 127.0.0.1, so they are not a secret.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "chatbot",
        "HOST": "postgres" if IS_DOCKER else "127.0.0.1",
        "PORT": 5432,
        "USER": "admin",
        "PASSWORD": "password",
        "CONN_MAX_AGE": 600,
        "CONN_HEALTH_CHECKS": True,
    }
}


LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

LOGGING = {
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
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "django_app": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "container": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
