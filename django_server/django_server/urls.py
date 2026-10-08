from django.http import JsonResponse
from django.urls import include, path

from django_app.openai_compat.views import ChatCompletionsView, EmbeddingsView, ModelsView


def health_check(request):
    return JsonResponse({"status": "ok"})


urlpatterns = [
    path("health/", health_check, name="health_check"),
    # Exact OpenAI path (no app-specific prefix) — clients set this server as
    # their OpenAI base URL and always request "/v1/chat/completions".
    path("v1/chat/completions", ChatCompletionsView.as_view(), name="openai_chat_completions"),
    path("v1/models", ModelsView.as_view(), name="openai_models"),
    path("v1/embeddings", EmbeddingsView.as_view(), name="openai_embeddings"),
    path("", include("django_app.urls")),
    path("docker/", include("container.urls")),
]
