from django.urls import path
from django.urls.resolvers import URLPattern

from . import views

urlpatterns: list[URLPattern] = [
    path("", views.Docker.as_view(), name="docker"),
    path("containers/", views.Containers.as_view(), name="containers"),
    path("hardware/", views.Hardware.as_view(), name="hardware"),
    path("disk-usage/", views.DiskUsage.as_view(), name="disk-usage"),
    path("ollama-image/", views.OllamaImage.as_view(), name="ollama-image"),
    path("container/<str:model>", views.Container.as_view(), name="container"),
    path("container/<str:model>/unload", views.ContainerUnload.as_view(), name="container-unload"),
    path("container/<str:model>/update", views.ContainerUpdate.as_view(), name="container-update"),
    path("runtime/", views.Runtime.as_view(), name="runtime"),
    path("ollama-update/", views.OllamaUpdate.as_view(), name="ollama-update"),
]
