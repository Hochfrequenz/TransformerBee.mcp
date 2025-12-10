"""Tests for EDIFACT summarization."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from transformerbeemcp.summarizer import summarize_edifact, _SYSTEM_PROMPT


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