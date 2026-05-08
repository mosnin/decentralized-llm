"""Tests that validate the Docker configuration files without running Docker."""

from __future__ import annotations

import stat
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent


def test_docker_entrypoint_is_executable() -> None:
    """The docker_entrypoint.sh script must exist and be executable."""
    entrypoint = PROJECT_ROOT / "scripts" / "docker_entrypoint.sh"
    assert entrypoint.exists(), f"Expected {entrypoint} to exist"
    file_stat = entrypoint.stat()
    is_executable = bool(file_stat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    assert is_executable, f"{entrypoint} must have execute permission set"


def test_dockerfile_node_has_healthcheck() -> None:
    """Dockerfile.node must include a HEALTHCHECK instruction."""
    dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.node"
    assert dockerfile.exists(), f"Expected {dockerfile} to exist"
    content = dockerfile.read_text(encoding="utf-8")
    assert "HEALTHCHECK" in content, "Dockerfile.node must contain a HEALTHCHECK instruction"


def test_dockerfile_api_exists() -> None:
    """docker/Dockerfile.api must exist."""
    dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.api"
    assert dockerfile.exists(), f"Expected {dockerfile} to exist"


def test_docker_compose_has_api_service() -> None:
    """docker-compose.yml must define an 'api' service."""
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    assert compose_file.exists(), f"Expected {compose_file} to exist"
    compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    services = compose.get("services", {})
    assert "api" in services, (
        f"docker-compose.yml must have an 'api' service; found: {list(services)}"
    )


def test_docker_compose_all_services_have_healthcheck() -> None:
    """Every service in docker-compose.yml must declare a healthcheck."""
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    assert compose_file.exists(), f"Expected {compose_file} to exist"
    compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    services = compose.get("services", {})
    missing = [name for name, cfg in services.items() if "healthcheck" not in (cfg or {})]
    assert not missing, f"Services missing healthcheck: {missing}"


def test_dockerfile_node_has_cuda_version_arg() -> None:
    """Dockerfile.node must declare ARG CUDA_VERSION so the base image can be parameterised."""
    dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.node"
    content = dockerfile.read_text(encoding="utf-8")
    assert "ARG CUDA_VERSION" in content, "Dockerfile.node must declare ARG CUDA_VERSION"


def test_dockerfile_node_has_multistage_build() -> None:
    """Dockerfile.node must use a multi-stage build (builder + runtime stages)."""
    dockerfile = PROJECT_ROOT / "docker" / "Dockerfile.node"
    content = dockerfile.read_text(encoding="utf-8")
    assert " AS builder" in content, "Dockerfile.node must have a 'builder' stage"
    assert " AS runtime" in content, "Dockerfile.node must have a 'runtime' stage"


def test_docker_compose_node_services_have_restart_policy() -> None:
    """Node services must have restart: unless-stopped for production reliability."""
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    services = compose.get("services", {})
    node_services = {k: v for k, v in services.items() if k.startswith("node-")}
    missing = [
        name
        for name, cfg in node_services.items()
        if (cfg or {}).get("restart") != "unless-stopped"
    ]
    assert not missing, f"Node services missing restart policy: {missing}"


def test_docker_compose_api_service_uses_api_dockerfile() -> None:
    """The 'api' service in docker-compose.yml must reference Dockerfile.api."""
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    api_service = compose.get("services", {}).get("api", {})
    build_cfg = api_service.get("build", {})
    dockerfile = build_cfg.get("dockerfile", "")
    assert "Dockerfile.api" in dockerfile, (
        f"'api' service build.dockerfile must reference Dockerfile.api, got: {dockerfile!r}"
    )


def test_docker_entrypoint_waits_for_rpc() -> None:
    """docker_entrypoint.sh must contain retry logic for SOLANA_RPC_URL readiness."""
    entrypoint = PROJECT_ROOT / "scripts" / "docker_entrypoint.sh"
    content = entrypoint.read_text(encoding="utf-8")
    assert "SOLANA_RPC_URL" in content, "Entrypoint must reference SOLANA_RPC_URL"
    # Check that there is some retry/loop construct
    has_retry = "MAX_RETRIES" in content or "until " in content or "retry" in content.lower()
    assert has_retry, "Entrypoint must contain retry logic for RPC readiness"
