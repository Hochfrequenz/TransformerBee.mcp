"""Tests for REST API."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from transformerbeemcp.rest_api import app, _rate_limit_store, verify_token


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_rate_limit_store():
    """Clear rate limit store before each test."""
    _rate_limit_store.clear()
    yield
    _rate_limit_store.clear()


def test_health_endpoint_healthy(client):
    """Test health endpoint returns healthy when Ollama is available."""
    with patch("transformerbeemcp.rest_api.check_ollama_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = {
            "ollama_host": "http://localhost:11434",
            "ollama_reachable": True,
            "model": "llama3",
            "model_available": True,
            "error": None,
        }

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["ollama_reachable"] is True
        assert data["model_available"] is True
        assert data["error"] is None


def test_health_endpoint_unhealthy_ollama_unreachable(client):
    """Test health endpoint returns unhealthy when Ollama is not reachable."""
    with patch("transformerbeemcp.rest_api.check_ollama_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = {
            "ollama_host": "http://localhost:11434",
            "ollama_reachable": False,
            "model": "llama3",
            "model_available": False,
            "error": "Cannot connect to Ollama",
        }

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["ollama_reachable"] is False
        assert data["error"] == "Cannot connect to Ollama"


def test_health_endpoint_unhealthy_model_not_found(client):
    """Test health endpoint returns unhealthy when model is not available."""
    with patch("transformerbeemcp.rest_api.check_ollama_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = {
            "ollama_host": "http://localhost:11434",
            "ollama_reachable": True,
            "model": "llama3",
            "model_available": False,
            "error": "Model 'llama3' not found",
        }

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["ollama_reachable"] is True
        assert data["model_available"] is False
        assert "not found" in data["error"]


def test_summarize_no_auth(client):
    """Test summarize endpoint requires authentication."""
    response = client.post("/summarize", json={"edifact": "UNB+..."})
    # FastAPI returns 401 when no credentials are provided
    assert response.status_code == 401


def test_summarize_invalid_token(client):
    """Test summarize endpoint rejects invalid token."""
    response = client.post(
        "/summarize",
        json={"edifact": "UNB+..."},
        headers={"Authorization": "Bearer invalid_token"},
    )
    assert response.status_code == 401


def test_summarize_success(client):
    """Test successful summarization with valid token."""
    mock_payload = {"sub": "user123", "aud": "https://transformer.bee"}

    # Override the dependency properly
    app.dependency_overrides[verify_token] = lambda: mock_payload

    with patch("transformerbeemcp.rest_api.summarize_edifact", new_callable=AsyncMock) as mock_summarize:
        mock_summarize.return_value = "Dies ist eine Testzusammenfassung."

        response = client.post(
            "/summarize",
            json={"edifact": "UNB+UNOC:3+..."},
            headers={"Authorization": "Bearer valid_token"},
        )

        assert response.status_code == 200
        assert response.json() == {"summary": "Dies ist eine Testzusammenfassung."}
        mock_summarize.assert_called_once_with("UNB+UNOC:3+...")

    app.dependency_overrides.clear()


def test_summarize_rate_limit(client):
    """Test rate limiting kicks in after too many requests."""
    mock_payload = {"sub": "user456", "aud": "https://transformer.bee"}

    app.dependency_overrides[verify_token] = lambda: mock_payload

    with patch("transformerbeemcp.rest_api.summarize_edifact", new_callable=AsyncMock) as mock_summarize:
        mock_summarize.return_value = "Summary"

        # Make RATE_LIMIT requests (default is 10)
        # For testing, we'll just verify rate limiting works after exceeding limit
        for i in range(10):
            response = client.post(
                "/summarize",
                json={"edifact": "UNB+..."},
                headers={"Authorization": "Bearer valid_token"},
            )
            assert response.status_code == 200, f"Request {i+1} failed"

        # 11th request should be rate limited
        response = client.post(
            "/summarize",
            json={"edifact": "UNB+..."},
            headers={"Authorization": "Bearer valid_token"},
        )
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]

    app.dependency_overrides.clear()


def test_summarize_error_handling_connect_error(client):
    """Test error handling when Ollama is not reachable."""
    import httpx

    mock_payload = {"sub": "user789", "aud": "https://transformer.bee"}

    app.dependency_overrides[verify_token] = lambda: mock_payload

    with patch("transformerbeemcp.rest_api.summarize_edifact", new_callable=AsyncMock) as mock_summarize:
        mock_summarize.side_effect = httpx.ConnectError("Connection refused")

        response = client.post(
            "/summarize",
            json={"edifact": "UNB+..."},
            headers={"Authorization": "Bearer valid_token"},
        )

        assert response.status_code == 500
        assert "Cannot connect to Ollama" in response.json()["detail"]

    app.dependency_overrides.clear()


def test_summarize_error_handling_timeout(client):
    """Test error handling when Ollama times out."""
    import httpx

    mock_payload = {"sub": "user790", "aud": "https://transformer.bee"}

    app.dependency_overrides[verify_token] = lambda: mock_payload

    with patch("transformerbeemcp.rest_api.summarize_edifact", new_callable=AsyncMock) as mock_summarize:
        mock_summarize.side_effect = httpx.TimeoutException("Request timed out")

        response = client.post(
            "/summarize",
            json={"edifact": "UNB+..."},
            headers={"Authorization": "Bearer valid_token"},
        )

        assert response.status_code == 500
        assert "timed out" in response.json()["detail"]

    app.dependency_overrides.clear()


def test_summarize_empty_edifact(client):
    """Test validation of empty EDIFACT input."""
    mock_payload = {"sub": "user000", "aud": "https://transformer.bee"}

    app.dependency_overrides[verify_token] = lambda: mock_payload

    with patch("transformerbeemcp.rest_api.summarize_edifact", new_callable=AsyncMock) as mock_summarize:
        mock_summarize.return_value = "Empty message summary"

        response = client.post(
            "/summarize",
            json={"edifact": ""},
            headers={"Authorization": "Bearer valid_token"},
        )

        # Empty string is technically valid, let the LLM handle it
        assert response.status_code == 200

    app.dependency_overrides.clear()
