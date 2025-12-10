"""Integration tests for EDIFACT summarization using Ollama testcontainer.

These tests require Docker to be running and can be slow (need to pull model).
They are skipped by default in CI. To run:
    SKIP_INTEGRATION_TESTS=false tox -e tests
"""

import os

import pytest


# Try to import testcontainers - skip tests if not installed
try:
    from testcontainers.ollama import OllamaContainer

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    OllamaContainer = None  # type: ignore

# Skip all tests in this module if testcontainers is not available or Docker is not running
pytestmark = [
    pytest.mark.skipif(
        not TESTCONTAINERS_AVAILABLE,
        reason="testcontainers not installed (install with: pip install transformerbeemcp[tests])",
    ),
    pytest.mark.skipif(
        os.getenv("SKIP_INTEGRATION_TESTS", "true").lower() == "true",
        reason="Integration tests skipped by default (set SKIP_INTEGRATION_TESTS=false to run)",
    ),
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def ollama_container():
    """Start Ollama container and pull a small model for testing."""
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not available")

    # Use tinyllama for faster tests (smaller model)
    with OllamaContainer("ollama/ollama:latest") as container:
        # Pull tinyllama - much smaller than llama3
        container.exec(["ollama", "pull", "tinyllama"])
        yield container


@pytest.fixture(scope="module")
def ollama_host(ollama_container):
    """Get the Ollama host URL from the container."""
    return ollama_container.get_endpoint()


@pytest.fixture
def set_ollama_env(ollama_host, monkeypatch):
    """Set environment variables for Ollama."""
    monkeypatch.setenv("OLLAMA_HOST", ollama_host)
    monkeypatch.setenv("OLLAMA_MODEL", "tinyllama")


@pytest.mark.anyio
async def test_check_ollama_health_with_container(set_ollama_env):
    """Test that health check works with real Ollama container."""
    # Import after setting env vars
    from transformerbeemcp.summarizer import check_ollama_health

    result = await check_ollama_health()

    assert result["ollama_reachable"] is True
    assert result["model_available"] is True
    assert result["error"] is None


@pytest.mark.anyio
async def test_check_ollama_health_model_not_found(ollama_host, monkeypatch):
    """Test health check when model is not available."""
    monkeypatch.setenv("OLLAMA_HOST", ollama_host)
    monkeypatch.setenv("OLLAMA_MODEL", "nonexistent-model-12345")

    # Need to reimport to pick up new env vars
    import importlib

    import transformerbeemcp.summarizer

    importlib.reload(transformerbeemcp.summarizer)
    from transformerbeemcp.summarizer import check_ollama_health

    result = await check_ollama_health()

    assert result["ollama_reachable"] is True
    assert result["model_available"] is False
    assert "not found" in result["error"].lower()


@pytest.mark.anyio
async def test_summarize_edifact_with_container(set_ollama_env):
    """Test actual EDIFACT summarization with real Ollama container."""
    # Import after setting env vars
    import importlib

    import transformerbeemcp.summarizer

    importlib.reload(transformerbeemcp.summarizer)
    from transformerbeemcp.summarizer import summarize_edifact

    # Simple EDIFACT-like message
    edifact = "UNB+UNOC:3+9904321000019:500+9900123000003:500+241210:1245+ABC123456789++TL'"

    # This will actually call Ollama - may be slow
    summary = await summarize_edifact(edifact, timeout=60.0)

    # Just check we got some response
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_health_endpoint_with_container(set_ollama_env):
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
    assert data["status"] == "healthy"
    assert data["ollama_reachable"] is True
    assert data["model_available"] is True
    assert data["error"] is None
