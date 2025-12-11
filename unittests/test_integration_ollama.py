"""Integration tests for EDIFACT summarization using Ollama testcontainer.

These tests require Docker to be running and can be slow (need to pull model).
To run: tox -e tests
"""

from typing import Generator

import pytest

# testcontainers is an optional dependency - skip tests if not installed
pytest.importorskip("testcontainers")
from testcontainers.ollama import OllamaContainer  # type: ignore[import-untyped]

from transformerbeemcp.rest_api import HealthResponse


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def ollama_container() -> Generator[OllamaContainer, None, None]:
    """Start Ollama container and pull a small model for testing."""
    # Use tinyllama for faster tests (smaller model)
    with OllamaContainer("ollama/ollama:latest") as container:
        # Pull tinyllama - much smaller than llama3
        container.exec(["ollama", "pull", "tinyllama"])
        yield container


@pytest.fixture(scope="module")
def ollama_host(ollama_container: OllamaContainer) -> str:
    """Get the Ollama host URL from the container."""
    return ollama_container.get_endpoint()  # type: ignore[no-any-return]


@pytest.fixture
def set_ollama_env(ollama_host: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set environment variables for Ollama."""
    monkeypatch.setenv("OLLAMA_HOST", ollama_host)
    monkeypatch.setenv("OLLAMA_MODEL", "tinyllama")


@pytest.mark.anyio
async def test_check_ollama_health_with_container(set_ollama_env: None) -> None:
    """Test that health check works with real Ollama container."""
    # Need to reload the module to pick up the env vars set by set_ollama_env fixture.
    # The module-level constants (_OLLAMA_HOST, _OLLAMA_MODEL) are evaluated at import time,
    # so we must reload after the env vars are set.
    import importlib

    import transformerbeemcp.summarizer

    importlib.reload(transformerbeemcp.summarizer)
    from transformerbeemcp.summarizer import check_ollama_health

    result = await check_ollama_health()

    assert result.ollama_reachable is True
    assert result.model_available is True
    assert result.error is None


@pytest.mark.anyio
async def test_check_ollama_health_model_not_found(ollama_host: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test health check when model is not available."""
    monkeypatch.setenv("OLLAMA_HOST", ollama_host)
    monkeypatch.setenv("OLLAMA_MODEL", "nonexistent-model-12345")

    # Need to reimport to pick up new env vars
    import importlib

    import transformerbeemcp.summarizer

    importlib.reload(transformerbeemcp.summarizer)
    from transformerbeemcp.summarizer import check_ollama_health

    result = await check_ollama_health()

    assert result.ollama_reachable is True
    assert result.model_available is False
    assert result.error is not None
    assert "not found" in result.error.lower()


@pytest.mark.anyio
async def test_summarize_bo4e_with_container(set_ollama_env: None) -> None:
    """Test actual BO4E summarization with real Ollama container."""
    # Import after setting env vars
    import importlib

    import transformerbeemcp.summarizer

    importlib.reload(transformerbeemcp.summarizer)
    from transformerbeemcp.summarizer import summarize_bo4e

    # Simple BO4E-like JSON message
    bo4e_json = '[{"typ": "Marktnachricht", "absender": "9904321000019", "empfaenger": "9900123000003"}]'

    # This will actually call Ollama - may be slow
    summary = await summarize_bo4e(bo4e_json, timeout=60.0)

    # Just check we got some response
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_health_endpoint_with_container(set_ollama_env: None) -> None:
    """Test REST API health endpoint with real Ollama container."""
    import importlib

    import transformerbeemcp.summarizer

    importlib.reload(transformerbeemcp.summarizer)

    # Also need to reimport rest_api to pick up the reloaded summarizer
    import transformerbeemcp.rest_api

    importlib.reload(transformerbeemcp.rest_api)

    from fastapi.testclient import TestClient

    from transformerbeemcp.rest_api import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    health_response = HealthResponse.model_validate(data)
    assert health_response.status == "healthy"
    assert health_response.ollama_reachable is True
    assert health_response.model_available is True
    assert health_response.error is None
