"""Tests for EDIFACT summarization.

Note on test isolation:
    The integration tests (test_integration_ollama.py) use importlib.reload() to reload
    the summarizer module with different environment variables (OLLAMA_HOST, OLLAMA_MODEL).
    This modifies the module-level constants _OLLAMA_HOST and _OLLAMA_MODEL which persist
    across test runs within the same pytest session.

    If integration tests run before these unit tests (alphabetical order: test_integration_*
    comes before test_summarizer.*), the unit tests would see "tinyllama" instead of "llama3"
    as the default model, causing assertion failures.

    The reset_summarizer_module fixture below resets these constants to their default values
    before each test to ensure test isolation regardless of execution order.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import transformerbeemcp.summarizer
from transformerbeemcp.summarizer import _SYSTEM_PROMPT


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_summarizer_module() -> None:
    """Reset the summarizer module constants to default values before each test.

    This ensures tests are isolated from side effects of integration tests that
    may have reloaded the module with different env vars (e.g., OLLAMA_MODEL=tinyllama).

    Without this fixture, tests would fail with assertions like:
        AssertionError: assert 'tinyllama' == 'llama3'

    because the integration tests modify module-level constants that persist across tests.
    """
    transformerbeemcp.summarizer._OLLAMA_HOST = "http://localhost:11434"
    transformerbeemcp.summarizer._OLLAMA_MODEL = "llama3"


@pytest.mark.anyio
async def test_summarize_edifact_success() -> None:
    """Test successful summarization."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Dies ist eine Testzusammenfassung."}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

        # Re-import to get the functions that use the reset module constants
        from transformerbeemcp.summarizer import summarize_edifact

        result = await summarize_edifact("UNB+UNOC:3+...")

        assert result == "Dies ist eine Testzusammenfassung."
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/api/generate" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "llama3"
        assert call_args[1]["json"]["system"] == _SYSTEM_PROMPT
        assert call_args[1]["json"]["prompt"] == "UNB+UNOC:3+..."
        assert call_args[1]["json"]["stream"] is False


@pytest.mark.anyio
async def test_summarize_edifact_http_error() -> None:
    """Test handling of HTTP errors from Ollama."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.HTTPStatusError(
        "Internal Server Error",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

        from transformerbeemcp.summarizer import summarize_edifact

        with pytest.raises(httpx.HTTPStatusError):
            await summarize_edifact("UNB+UNOC:3+...")


@pytest.mark.anyio
async def test_summarize_edifact_connection_error() -> None:
    """Test handling of connection errors to Ollama."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

        from transformerbeemcp.summarizer import summarize_edifact

        with pytest.raises(httpx.ConnectError):
            await summarize_edifact("UNB+UNOC:3+...")


@pytest.mark.anyio
async def test_check_ollama_health_success() -> None:
    """Test health check when Ollama is reachable and model is available."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {"name": "llama3:latest"},
            {"name": "tinyllama:latest"},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

        # Import both function and class together to ensure they're from the same module state
        from transformerbeemcp.summarizer import OllamaHealthStatus as HealthStatus
        from transformerbeemcp.summarizer import check_ollama_health

        result = await check_ollama_health()

        assert isinstance(result, HealthStatus)
        assert result.ollama_reachable is True
        assert result.model_available is True
        assert result.error is None
        mock_client.get.assert_called_once()


@pytest.mark.anyio
async def test_check_ollama_health_model_not_found() -> None:
    """Test health check when model is not available."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {"name": "tinyllama:latest"},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

        # Import both function and class together to ensure they're from the same module state
        from transformerbeemcp.summarizer import OllamaHealthStatus as HealthStatus
        from transformerbeemcp.summarizer import check_ollama_health

        result = await check_ollama_health()

        assert isinstance(result, HealthStatus)
        assert result.ollama_reachable is True
        assert result.model_available is False
        assert result.error is not None
        assert "not found" in result.error.lower()


@pytest.mark.anyio
async def test_check_ollama_health_connection_error() -> None:
    """Test health check when Ollama is not reachable."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

        # Import both function and class together to ensure they're from the same module state
        from transformerbeemcp.summarizer import OllamaHealthStatus as HealthStatus
        from transformerbeemcp.summarizer import check_ollama_health

        result = await check_ollama_health()

        assert isinstance(result, HealthStatus)
        assert result.ollama_reachable is False
        assert result.model_available is False
        assert result.error is not None
        assert "connect" in result.error.lower()
