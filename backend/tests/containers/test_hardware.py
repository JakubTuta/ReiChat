"""get_gpu_vram_bytes never touches a real GPU or Docker daemon here — nvidia-smi
and the container manager are both mocked.
"""

import subprocess
from unittest.mock import MagicMock

from app.containers import hardware


def test_get_gpu_vram_bytes_returns_none_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)

    assert hardware.get_gpu_vram_bytes() is None


def test_get_gpu_vram_bytes_parses_mib_to_bytes(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: MagicMock(stdout="8192\n"))

    assert hardware.get_gpu_vram_bytes() == 8192 * 1024 * 1024


def test_get_gpu_vram_bytes_returns_none_when_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def raise_failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(hardware.subprocess, "run", raise_failed)

    assert hardware.get_gpu_vram_bytes() is None


def test_get_gpu_vram_bytes_returns_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: MagicMock(stdout="not-a-number\n"))

    assert hardware.get_gpu_vram_bytes() is None


def test_get_gpu_vram_bytes_returns_none_on_timeout(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(hardware.subprocess, "run", raise_timeout)

    assert hardware.get_gpu_vram_bytes() is None


async def test_vram_via_ollama_container_parses_what_nvidia_smi_printed_there():
    manager = MagicMock()

    async def exec_in_gpu_container(command):
        assert command == hardware.NVIDIA_SMI_QUERY
        return "12288\n"

    manager.exec_in_gpu_container = exec_in_gpu_container

    assert await hardware.get_gpu_vram_bytes_via_ollama_container(manager) == 12288 * 1024 * 1024


async def test_vram_via_ollama_container_is_none_without_a_gpu_container():
    manager = MagicMock()

    async def exec_in_gpu_container(command):
        return None

    manager.exec_in_gpu_container = exec_in_gpu_container

    assert await hardware.get_gpu_vram_bytes_via_ollama_container(manager) is None


async def test_vram_via_ollama_container_is_none_when_nvidia_smi_prints_nonsense():
    manager = MagicMock()

    async def exec_in_gpu_container(command):
        return "N/A\n"

    manager.exec_in_gpu_container = exec_in_gpu_container

    assert await hardware.get_gpu_vram_bytes_via_ollama_container(manager) is None


async def test_detect_vram_prefers_the_host_gpu(monkeypatch):
    async def never(manager):
        raise AssertionError("asked the container although the host answered")

    monkeypatch.setattr(hardware, "get_gpu_vram_bytes", lambda: 8192)
    monkeypatch.setattr(hardware, "get_gpu_vram_bytes_via_ollama_container", never)

    assert await hardware.detect_vram_bytes(MagicMock()) == 8192


async def test_detect_vram_falls_back_to_the_container_and_caches(monkeypatch):
    monkeypatch.setattr(hardware, "get_gpu_vram_bytes", lambda: None)
    calls = []

    async def via_container(manager):
        calls.append(manager)
        return 4096

    monkeypatch.setattr(hardware, "get_gpu_vram_bytes_via_ollama_container", via_container)

    assert await hardware.detect_vram_bytes(MagicMock()) == 4096
    assert await hardware.detect_vram_bytes(MagicMock()) == 4096
    assert len(calls) == 1
