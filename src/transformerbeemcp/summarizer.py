"""EDIFACT summarization using Ollama."""

import logging
import os

import httpx

_logger = logging.getLogger(__name__)

# Ollama configuration (module-private, not intended for external import)
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

_SYSTEM_PROMPT = """Du bist ein Experte für EDIFACT-Nachrichten im deutschen Energiemarkt.
Fasse die folgende EDIFACT-Nachricht in einfachem Deutsch zusammen.
Erkläre den Nachrichtentyp, die beteiligten Parteien, und die wesentlichen Inhalte.
Antworte präzise und verständlich für Sachbearbeiter ohne EDIFACT-Kenntnisse."""


async def check_ollama_health(timeout: float = 5.0) -> dict:
    """
    Check if Ollama is reachable and the configured model is available.

    Args:
        timeout: Request timeout in seconds

    Returns:
        dict with keys:
        - ollama_host: configured host URL
        - ollama_reachable: bool
        - model: configured model name
        - model_available: bool
        - error: error message if any check failed
    """
    result = {
        "ollama_host": _OLLAMA_HOST,
        "ollama_reachable": False,
        "model": _OLLAMA_MODEL,
        "model_available": False,
        "error": None,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Check if Ollama is reachable via /api/tags endpoint
            response = await client.get(f"{_OLLAMA_HOST}/api/tags")
            response.raise_for_status()
            result["ollama_reachable"] = True

            # Check if the configured model is available
            data = response.json()
            available_models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            # Also check full name with tag (e.g., "llama3:latest")
            available_models_full = [m.get("name", "") for m in data.get("models", [])]

            if _OLLAMA_MODEL in available_models or _OLLAMA_MODEL in available_models_full:
                result["model_available"] = True
            else:
                result["error"] = f"Model '{_OLLAMA_MODEL}' not found. Available: {available_models_full}"

    except httpx.ConnectError as e:
        result["error"] = f"Cannot connect to Ollama at {_OLLAMA_HOST}: {e}"
    except httpx.HTTPStatusError as e:
        result["ollama_reachable"] = True  # It responded, just with an error
        result["error"] = f"Ollama returned error: {e.response.status_code}"
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"

    return result


async def summarize_edifact(edifact: str, timeout: float = 120.0) -> str:
    """
    Generate a German summary of an EDIFACT message using Ollama.

    Args:
        edifact: Raw EDIFACT message string
        timeout: Request timeout in seconds (default 120s for LLM inference)

    Returns:
        Human-readable German summary

    Raises:
        httpx.HTTPStatusError: If Ollama returns an error response
        httpx.ConnectError: If Ollama is not reachable
    """
    _logger.info("Summarizing EDIFACT message using model '%s' at '%s'", _OLLAMA_MODEL, _OLLAMA_HOST)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{_OLLAMA_HOST}/api/generate",
            json={
                "model": _OLLAMA_MODEL,
                "system": _SYSTEM_PROMPT,
                "prompt": edifact,
                "stream": False,
            },
        )
        response.raise_for_status()
        result = response.json()
        summary = result["response"]
        _logger.debug("Generated summary: %s", summary[:100] + "..." if len(summary) > 100 else summary)
        return summary
