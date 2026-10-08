"""get_host_ram_bytes/get_gpu_vram_bytes never touch a real Docker daemon or
GPU here — both external calls (docker client, nvidia-smi) are mocked.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from container import hardware


def test_get_host_ram_bytes_reads_mem_total_from_docker_info():
    fake_client = MagicMock()
    fake_client.info.return_value = {"MemTotal": 16_000_000_000}
    fake_manager = MagicMock(client=fake_client)

    assert hardware.get_host_ram_bytes(fake_manager) == 16_000_000_000


def test_get_host_ram_bytes_returns_none_when_client_is_none():
    fake_manager = MagicMock(client=None)

    assert hardware.get_host_ram_bytes(fake_manager) is None


def test_get_host_ram_bytes_returns_none_on_docker_error():
    fake_client = MagicMock()
    fake_client.info.side_effect = Exception("daemon unreachable")
    fake_manager = MagicMock(client=fake_client)

    assert hardware.get_host_ram_bytes(fake_manager) is None


def test_get_gpu_vram_bytes_returns_none_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)

    assert hardware.get_gpu_vram_bytes() is None


def test_get_gpu_vram_bytes_parses_mib_to_bytes(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    fake_result = MagicMock(stdout="8192\n")
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: fake_result)

    assert hardware.get_gpu_vram_bytes() == 8192 * 1024 * 1024


def test_get_gpu_vram_bytes_returns_none_when_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(hardware.subprocess, "run", _raise)

    assert hardware.get_gpu_vram_bytes() is None


def test_get_gpu_vram_bytes_returns_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    fake_result = MagicMock(stdout="not-a-number\n")
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: fake_result)

    assert hardware.get_gpu_vram_bytes() is None


@pytest.mark.parametrize("timeout_exc", [subprocess.TimeoutExpired("nvidia-smi", 5)])
def test_get_gpu_vram_bytes_returns_none_on_timeout(monkeypatch, timeout_exc):
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def _raise(*args, **kwargs):
        raise timeout_exc

    monkeypatch.setattr(hardware.subprocess, "run", _raise)

    assert hardware.get_gpu_vram_bytes() is None


def test_vram_via_ollama_container_runs_nvidia_smi_inside_a_gpu_container():
    gpu_container = MagicMock()
    gpu_container.exec_run.return_value = (0, b"12288\n")
    fake_client = MagicMock()
    fake_client.containers.list.return_value = [gpu_container]

    assert hardware.get_gpu_vram_bytes_via_ollama_container(MagicMock(client=fake_client)) == 12288 * 1024 * 1024
    assert "reichat.gpu=true" in fake_client.containers.list.call_args.kwargs["filters"]["label"]


def test_vram_via_ollama_container_is_none_without_a_gpu_container():
    fake_client = MagicMock()
    fake_client.containers.list.return_value = []

    assert hardware.get_gpu_vram_bytes_via_ollama_container(MagicMock(client=fake_client)) is None


def test_detect_vram_falls_back_to_the_container_and_caches(monkeypatch):
    monkeypatch.setattr(hardware, "_cached_vram_bytes", None)
    monkeypatch.setattr(hardware, "get_gpu_vram_bytes", lambda: None)
    calls = []
    monkeypatch.setattr(
        hardware, "get_gpu_vram_bytes_via_ollama_container", lambda cm: calls.append(cm) or 4096
    )

    assert hardware.detect_vram_bytes(MagicMock()) == 4096
    assert hardware.detect_vram_bytes(MagicMock()) == 4096
    assert len(calls) == 1
