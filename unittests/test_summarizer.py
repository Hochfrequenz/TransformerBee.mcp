"""Tests for EDIFACT summarization."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from transformerbeemcp.summarizer import check_ollama_health, summarize_edifact, _SYSTEM_PROMPT


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_summarize_edifact_success():
    """Test successful summarization."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Dies ist eine Testzusammenfassung."}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

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
async def test_summarize_edifact_http_error():
    """Test handling of HTTP errors from Ollama."""
    import httpx

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.HTTPStatusError(
        "Internal Server Error",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            await summarize_edifact("UNB+UNOC:3+...")


@pytest.mark.anyio
async def test_summarize_edifact_connection_error():
    """Test handling of connection errors to Ollama."""
    import httpx

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

        with pytest.raises(httpx.ConnectError):
            await summarize_edifact("UNB+UNOC:3+...")


@pytest.mark.anyio
async def test_check_ollama_health_success():
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

        result = await check_ollama_health()

        assert result["ollama_reachable"] is True
        assert result["model_available"] is True
        assert result["error"] is None
        mock_client.get.assert_called_once()


@pytest.mark.anyio
async def test_check_ollama_health_model_not_found():
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

        result = await check_ollama_health()

        assert result["ollama_reachable"] is True
        assert result["model_available"] is False
        assert "not found" in result["error"].lower()


@pytest.mark.anyio
async def test_check_ollama_health_connection_error():
    """Test health check when Ollama is not reachable."""
    import httpx

    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")

    with patch("transformerbeemcp.summarizer.httpx.AsyncClient") as mock_async_client:
        mock_async_client.return_value.__aenter__.return_value = mock_client

        result = await check_ollama_health()

        assert result["ollama_reachable"] is False
        assert result["model_available"] is False
        assert result["error"] is not None
        assert "connect" in result["error"].lower()