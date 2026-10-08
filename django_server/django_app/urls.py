from django.urls import path
from django.urls.resolvers import URLPattern

from . import views

urlpatterns: list[URLPattern] = [
    path(
        "chat-history/<str:model>/<str:chat_id>",
        views.ChatHistory.as_view(),
        name="chat_history",
    ),
    path("all-chats/<str:model>", views.AllChats.as_view(), name="all_chats"),
    path("chat-continue/<str:model>", views.ContinueChat.as_view(), name="chat_continue"),
    path("chat-context/<str:model>/<str:chat_id>", views.ChatContext.as_view(), name="chat_context"),
    path("ai-models/", views.AIModels.as_view(), name="ai_models"),
    path("personas/", views.Personas.as_view(), name="personas"),
    path("prompt-templates/", views.PromptTemplates.as_view(), name="prompt_templates"),
    path("search/<str:model>", views.Search.as_view(), name="search"),
    path("collections/", views.Collections.as_view(), name="collections"),
    path("documents/", views.Documents.as_view(), name="documents"),
    path("documents/chunk/", views.DocumentChunk.as_view(), name="document_chunk"),
    path("mcp-servers/", views.MCPServers.as_view(), name="mcp_servers"),
    path("mcp-servers/test/", views.MCPServerTest.as_view(), name="mcp_server_test"),
]
