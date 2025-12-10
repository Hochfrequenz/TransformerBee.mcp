"""REST API for EDIFACT summarization."""

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from pydantic import BaseModel

from transformerbeemcp.summarizer import OllamaHealthStatus, check_ollama_health, summarize_edifact

_logger = logging.getLogger(__name__)

# Auth0 configuration (module-private, not intended for external import)
# Uses the same Auth0 tenant and audience as transformer.bee, allowing clients
# (e.g., marktnachrichten-dolmetscher) to reuse their existing access tokens
# for both transformer.bee API and this summarization endpoint.
_AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "hochfrequenz.eu.auth0.com")
_AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://transformer.bee")
_JWKS_URL = f"https://{_AUTH0_DOMAIN}/.well-known/jwks.json"

# CORS configuration (module-private)
_ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,https://nice-mushroom-04ebea203.3.azurestaticapps.net,https://thankful-water-00644131e.3.azurestaticapps.net",
).split(",")

# Rate limiting configuration (module-private)
_RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))
_RATE_WINDOW_SECONDS = int(os.getenv("RATE_WINDOW_SECONDS", "60"))
# Maps user_id (from JWT 'sub' claim) -> list of request timestamps (UTC datetime)
_rate_limit_store: dict[str, list[datetime]] = defaultdict(list)

def _get_version() -> str:
    """Get version from package metadata."""
    try:
        return version("transformerbeemcp")
    except PackageNotFoundError:
        # Fallback for development or when package is not installed
        return "0.0.0-dev"


app = FastAPI(
    title="TransformerBee Summarizer",
    description="REST API for EDIFACT summarization using local LLM",
    version=_get_version(),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

security = HTTPBearer()
jwks_client: PyJWKClient | None = None


def get_jwks_client() -> PyJWKClient:
    """Lazy initialization of JWKS client."""
    global jwks_client  # pylint: disable=global-statement
    if jwks_client is None:
        jwks_client = PyJWKClient(_JWKS_URL)
    return jwks_client


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict[str, Any]:
    """
    Verify Auth0 JWT token.

    Returns:
        JWT payload dict containing at minimum:
        - sub: str - Subject identifier (user ID)
        - aud: str | list[str] - Audience claim
        - iss: str - Issuer URL
        - exp: int - Expiration timestamp
        - iat: int - Issued at timestamp
    """
    token = credentials.credentials
    try:
        signing_key = get_jwks_client().get_signing_key_from_jwt(token)
        payload: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=_AUTH0_AUDIENCE,
            issuer=f"https://{_AUTH0_DOMAIN}/",
        )
        return payload
    except jwt.exceptions.InvalidTokenError as e:
        _logger.warning("Invalid token: %s", e)
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e


def check_rate_limit(user_id: str) -> None:
    """
    Check if user has exceeded rate limit.

    Args:
        user_id: The user identifier (typically from JWT 'sub' claim)

    Raises:
        HTTPException: 429 status if rate limit is exceeded
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=_RATE_WINDOW_SECONDS)
    # Remove old entries outside the window
    _rate_limit_store[user_id] = [t for t in _rate_limit_store[user_id] if t > window_start]

    if len(_rate_limit_store[user_id]) >= _RATE_LIMIT:
        _logger.warning("Rate limit exceeded for user %s", user_id)
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Max {_RATE_LIMIT} requests per minute.")

    _rate_limit_store[user_id].append(now)


class SummarizeRequest(BaseModel):
    """Request body for summarization endpoint."""

    edifact: str


class SummarizeResponse(BaseModel):
    """Response body for summarization endpoint."""

    summary: str


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str
    detail: str | None = None


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid or missing authentication token"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Summarization failed"},
    },
)
async def summarize(request: SummarizeRequest, token_payload: dict[str, Any] = Depends(verify_token)) -> SummarizeResponse:
    """
    Generate a German summary of an EDIFACT message.

    Requires a valid Auth0 bearer token. Rate limited to 10 requests per minute per user.
    """
    user_id = token_payload.get("sub", "anonymous")
    check_rate_limit(user_id)

    _logger.info("Summarization requested by user %s", user_id)

    try:
        summary = await summarize_edifact(request.edifact)
        return SummarizeResponse(summary=summary)
    except httpx.HTTPStatusError as e:
        _logger.exception("Ollama returned error status")
        raise HTTPException(status_code=500, detail=f"Ollama error: {e.response.status_code}") from e
    except httpx.ConnectError as e:
        _logger.exception("Cannot connect to Ollama")
        raise HTTPException(status_code=500, detail=f"Cannot connect to Ollama: {e}") from e
    except httpx.TimeoutException as e:
        _logger.exception("Ollama request timed out")
        raise HTTPException(status_code=500, detail=f"Ollama request timed out: {e}") from e


class HealthResponse(BaseModel):
    """Health check response body."""

    status: str
    ollama_host: str
    ollama_reachable: bool
    model: str
    model_available: bool
    error: str | None = None


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint (no auth required).

    Checks if Ollama is reachable and the configured model is available.
    Returns status "healthy" only if all checks pass.
    """
    ollama_status = await check_ollama_health()

    is_healthy = ollama_status.ollama_reachable and ollama_status.model_available

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        ollama_host=ollama_status.ollama_host,
        ollama_reachable=ollama_status.ollama_reachable,
        model=ollama_status.model,
        model_available=ollama_status.model_available,
        error=ollama_status.error,
    )


def main() -> None:
    """
    CLI entry point for REST API server.

    This is used by the `run-transformerbee-rest-api` console script (defined in pyproject.toml).
    Docker uses `fastapi run` instead, which directly imports the `app` object.
    """
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    _logger.info("Starting REST API server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
