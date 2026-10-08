from collections.abc import Callable

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


class OriginGuardMiddleware:
    """Rejects state-changing requests sent by a browser page on any other
    origin. There is no auth, so DRF never enforces CSRF (only its
    SessionAuthentication does), and CORS alone doesn't stop a write: a
    form-encoded or multipart POST is a "simple request" that the browser
    sends without a preflight — only the *response* is hidden from the page.
    Any website the user visits could otherwise blind-POST e.g. a stdio MCP
    server whose command then runs in the backend container, which holds the
    Docker socket (= host root).

    Browsers always attach `Origin` to cross-origin POST/PUT/PATCH/DELETE, so
    a missing header means a non-browser client (curl, the OpenAI SDK against
    /v1/), which a web page can't impersonate. `Origin: null` (sandboxed
    iframes, file://) is rejected like any other foreign origin.
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response
        self.allowed_origins = frozenset(settings.CORS_ALLOWED_ORIGINS)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        origin = request.headers.get("Origin")

        if (
            request.method not in _SAFE_METHODS
            and origin is not None
            and origin not in self.allowed_origins
        ):
            return JsonResponse(
                {"error": "Requests from other websites are not allowed."}, status=403
            )

        return self.get_response(request)
