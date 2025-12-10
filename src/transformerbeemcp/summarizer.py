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
