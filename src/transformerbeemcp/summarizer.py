"""EDIFACT summarization using Ollama.

Converts EDIFACT to BO4E via transformer.bee before sending to Ollama.
"""

import json
import logging
import os

import aiohttp
import httpx
from efoli import EdifactFormatVersion, get_current_edifact_format_version
from pydantic import BaseModel
from transformerbeeclient.models.marktnachricht import Marktnachricht
from transformerbeeclient.models.transformerapi import EdifactToBo4eRequest, EdifactToBo4eResponse
from yarl import URL

_logger = logging.getLogger(__name__)


class OllamaHealthStatus(BaseModel):
    """Health status of Ollama connection and model availability."""

    ollama_host: str
    ollama_reachable: bool
    model: str
    model_available: bool
    error: str | None = None


# Ollama configuration (module-private, not intended for external import)
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Transformer.bee configuration (module-private)
_TRANSFORMERBEE_HOST = os.getenv("TRANSFORMERBEE_HOST", "https://transformerstage.utilibee.io")

_SYSTEM_PROMPT = """Du bist ein Experte für BO4E (Business Objects for Energy) im deutschen Energiemarkt.
Fasse die folgende BO4E-Nachricht in einfachem Deutsch zusammen.
Erkläre den Nachrichtentyp, die beteiligten Parteien, und die wesentlichen Inhalte.
Antworte präzise und verständlich für Sachbearbeiter ohne technische Kenntnisse."""


async def check_ollama_health(timeout: float = 5.0) -> OllamaHealthStatus:
    """
    Check if Ollama is reachable and the configured model is available.

    Args:
        timeout: Request timeout in seconds

    Returns:
        OllamaHealthStatus with connection and model availability information
    """
    ollama_reachable = False
    model_available = False
    error: str | None = None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Check if Ollama is reachable via /api/tags endpoint
            response = await client.get(f"{_OLLAMA_HOST}/api/tags")
            response.raise_for_status()
            ollama_reachable = True

            # Check if the configured model is available
            data = response.json()
            available_models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            # Also check full name with tag (e.g., "llama3:latest")
            available_models_full = [m.get("name", "") for m in data.get("models", [])]

            if _OLLAMA_MODEL in available_models or _OLLAMA_MODEL in available_models_full:
                model_available = True
            else:
                error = f"Model '{_OLLAMA_MODEL}' not found. Available: {available_models_full}"

    except httpx.ConnectError as e:
        error = f"Cannot connect to Ollama at {_OLLAMA_HOST}: {e}"
    except httpx.HTTPStatusError as e:
        ollama_reachable = True  # It responded, just with an error
        error = f"Ollama returned error: {e.response.status_code}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Catch any unexpected errors to ensure health check always returns a valid response
        error = f"Unexpected error: {e}"

    return OllamaHealthStatus(
        ollama_host=_OLLAMA_HOST,
        ollama_reachable=ollama_reachable,
        model=_OLLAMA_MODEL,
        model_available=model_available,
        error=error,
    )


async def convert_edifact_to_bo4e(
    edifact: str,
    auth_token: str,
    edifact_format_version: EdifactFormatVersion | None = None,
    timeout: float = 30.0,
) -> str:
    """
    Convert an EDIFACT message to BO4E JSON using transformer.bee.

    Uses the transformerbeeclient models for request/response handling,
    but makes the HTTP call directly to support forwarding an externally-obtained token.

    TODO: Once transformerbeeclient supports passing an externally-obtained bearer token
    (instead of only OAuth client credentials flow), refactor this to use the library's
    client directly. See: https://github.com/Hochfrequenz/TransformerBeeClient.py

    Args:
        edifact: Raw EDIFACT message string
        auth_token: Bearer token to forward to transformer.bee
        edifact_format_version: EDIFACT format version (defaults to current)
        timeout: Request timeout in seconds

    Returns:
        BO4E JSON string (serialized list of Marktnachricht)

    Raises:
        ValueError: If TRANSFORMERBEE_HOST is not configured
        aiohttp.ClientResponseError: If transformer.bee returns an error
    """
    if not _TRANSFORMERBEE_HOST:
        raise ValueError("TRANSFORMERBEE_HOST environment variable is not set")

    if edifact_format_version is None:
        edifact_format_version = get_current_edifact_format_version()

    _logger.info("Converting EDIFACT to BO4E using transformer.bee at '%s'", _TRANSFORMERBEE_HOST)

    base_url = URL(_TRANSFORMERBEE_HOST)
    edi_to_bo4e_url = base_url / "v1" / "transformer" / "EdiToBo4E"
    request = EdifactToBo4eRequest(edifact=edifact, format_version=edifact_format_version)  # type: ignore[call-arg]
    headers = {"Authorization": f"Bearer {auth_token}"}

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), raise_for_status=True) as session:
        async with session.post(
            url=edi_to_bo4e_url, json=request.model_dump(by_alias=True), headers=headers
        ) as response:
            response_json = await response.json()
            response_model = EdifactToBo4eResponse.model_validate(response_json)
            marktnachrichten = [
                Marktnachricht.model_validate(x) for x in json.loads(response_model.bo4e_json.replace("\\n", "\n"))
            ]
            _logger.debug("Converted to BO4E: %d Marktnachricht(en)", len(marktnachrichten))
            # Serialize to JSON for the LLM
            return "[" + ",".join(m.model_dump_json() for m in marktnachrichten) + "]"


async def summarize_bo4e(bo4e_json: str, timeout: float = 120.0) -> str:
    """
    Generate a German summary of a BO4E message using Ollama.

    Args:
        bo4e_json: BO4E message as JSON string
        timeout: Request timeout in seconds (default 120s for LLM inference)

    Returns:
        Human-readable German summary

    Raises:
        httpx.HTTPStatusError: If Ollama returns an error response
        httpx.ConnectError: If Ollama is not reachable
    """
    _logger.info("Summarizing BO4E message using model '%s' at '%s'", _OLLAMA_MODEL, _OLLAMA_HOST)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{_OLLAMA_HOST}/api/generate",
            json={
                "model": _OLLAMA_MODEL,
                "system": _SYSTEM_PROMPT,
                "prompt": bo4e_json,
                "stream": False,
            },
        )
        response.raise_for_status()
        result = response.json()
        summary: str = result["response"]
        _logger.debug("Generated summary: %s", summary[:100] + "..." if len(summary) > 100 else summary)
        return summary


async def summarize_edifact(edifact: str, auth_token: str, timeout: float = 120.0) -> str:
    """
    Generate a German summary of an EDIFACT message using Ollama.

    First converts EDIFACT to BO4E via transformer.bee, then sends the BO4E JSON
    to Ollama for summarization.

    Args:
        edifact: Raw EDIFACT message string
        auth_token: Bearer token to forward to transformer.bee
        timeout: Request timeout in seconds (default 120s for LLM inference)

    Returns:
        Human-readable German summary

    Raises:
        httpx.HTTPStatusError: If Ollama returns an error response
        httpx.ConnectError: If Ollama is not reachable
        ValueError: If TRANSFORMERBEE_HOST is not configured
    """
    # Convert EDIFACT to BO4E first
    bo4e_json = await convert_edifact_to_bo4e(edifact, auth_token)
    return await summarize_bo4e(bo4e_json, timeout)
