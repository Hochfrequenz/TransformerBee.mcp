"""
MCP server that provides a tool to convert between EDIFACT and BO4E formats using the TransformerBeeClient.
"""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from aiohttp import ClientResponseError
from efoli import EdifactFormatVersion, get_current_edifact_format_version
from mcp.server.fastmcp import Context, FastMCP
from transformerbeeclient import (
    AuthenticatedTransformerBeeClient,
    BOneyComb,
    TransformerBeeClient,
    UnauthenticatedTransformerBeeClient,
)

from transformerbeemcp.summarizer import summarize_bo4e

_logger = logging.getLogger(__name__)

_HOST_KEY = "TRANSFORMERBEE_HOST"
_CLIENT_ID_KEY = "TRANSFORMERBEE_CLIENT_ID"
_CLIENT_SECRET_KEY = "TRANSFORMERBEE_CLIENT_SECRET"


@dataclass
class AppContext:
    """global context for the application"""

    transformerbeeclient: TransformerBeeClient


def create_client(host: str, client_id: str | None, client_secret: str | None) -> TransformerBeeClient:
    """create a new transformer.bee client"""
    if not client_id or not client_secret:
        _logger.info(
            "Environment variables '%s' and/or '%s' are not set, using unauthenticated client",
            _CLIENT_ID_KEY,
            _CLIENT_SECRET_KEY,
        )
        return UnauthenticatedTransformerBeeClient(host)
    _logger.info("Using authenticated client id '%s' and respective secret", client_id)
    return AuthenticatedTransformerBeeClient(
        host,
        oauth_client_id=client_id,
        oauth_client_secret=client_secret,
    )


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:  # pylint:disable=unused-argument
    """Manage application lifecycle with type-safe context"""
    _logger.info("Trying to get environment variables; Start with '%s'", _HOST_KEY)
    transformerbee_host: str | None = os.environ.get(_HOST_KEY, None)
    if not transformerbee_host:
        raise ValueError(f"Environment variable '{_HOST_KEY}' not set")

    _logger.info("Using host '%s'", transformerbee_host)
    transformerbee_client_id: str | None = os.environ.get(_CLIENT_ID_KEY, None)
    transformerbee_client_secret: str | None = os.environ.get(_CLIENT_SECRET_KEY, None)
    transformerbee_client = create_client(transformerbee_host, transformerbee_client_id, transformerbee_client_secret)
    try:
        _logger.info("Instantiating context")
        yield AppContext(transformerbeeclient=transformerbee_client)
    finally:
        if hasattr(transformerbee_client, "close_session"):
            await transformerbee_client.close_session()


mcp = FastMCP("TransformerBee.mcp", dependencies=["transformerbeeclient"], lifespan=app_lifespan)


@mcp.tool(description="Convert an EDIFACT message to its BO4E equivalent")
async def convert_edifact_to_bo4e(
    ctx: Context,  # type:ignore[type-arg] # no idea what the second type arg is
    edifact: str,
    edifact_format_version: EdifactFormatVersion | None = None,
) -> dict[str, Any]:
    """Tool that uses initialized resources"""
    _logger.debug("Context: %s", str(ctx.request_context.lifespan_context))
    client: TransformerBeeClient = ctx.request_context.lifespan_context.transformerbeeclient
    if not edifact_format_version:
        edifact_format_version = get_current_edifact_format_version()
    try:
        marktnachrichten = await client.convert_to_bo4e(edifact=edifact, edifact_format_version=edifact_format_version)
    except ClientResponseError as cre:
        _logger.warning("transformer.bee rejected the request %s: %s", cre.request_info, cre.message)
        _logger.exception(cre)
        raise
    except Exception:
        _logger.exception("Error while converting EDIFACT to BO4E")
        raise
    if len(marktnachrichten) > 1:
        raise NotImplementedError(f"More than 1 Marktnachricht (got {len(marktnachrichten)}) not support yet")
    marktnachricht = marktnachrichten[0]
    await ctx.info(f"Successfully converted Marktnachricht with UNH {marktnachricht.unh} to BO4E")
    if len(marktnachricht.transaktionen) > 1:
        raise NotImplementedError(f"More than 1 transaction (got {len(marktnachricht.transaktionen)}) not support yet")
    transaktion = marktnachricht.transaktionen[0]
    return transaktion.model_dump(mode="json")


@mcp.tool(description="Convert a BO4E transaktion to its EDIFACT equivalent")
async def convert_bo4e_to_edifact(
    ctx: Context,  # type:ignore[type-arg] # no idea what the second type arg is
    transaktion: BOneyComb,
    edifact_format_version: EdifactFormatVersion | None = None,
) -> str:
    """Tool that uses initialized resources"""
    if not edifact_format_version:
        edifact_format_version = get_current_edifact_format_version()
    client: TransformerBeeClient = ctx.request_context.lifespan_context.transformerbeeclient
    try:
        edifact = await client.convert_to_edifact(boney_comb=transaktion, edifact_format_version=edifact_format_version)
    except Exception:
        _logger.exception("Error while converting BO4E to edifact")
        raise
    await ctx.info(f"Successfully converted BO4E to EDIFACT with format version {edifact_format_version}")
    return edifact


@mcp.tool(description="Generate a human-readable German summary of an EDIFACT message using a local LLM")
async def summarize_edifact_message(
    ctx: Context,  # type:ignore[type-arg]
    edifact: str,
    edifact_format_version: EdifactFormatVersion | None = None,
) -> str:
    """
    Generate a human-readable German summary of an EDIFACT message.

    First converts the EDIFACT to BO4E via transformer.bee, then uses a local
    Ollama instance to analyze the BO4E and produce a summary explaining the
    message type, involved parties, and key content.
    The summary is written in German, suitable for staff without EDIFACT expertise.

    Args:
        edifact: Raw EDIFACT message string
        edifact_format_version: EDIFACT format version (defaults to current)

    Returns:
        German summary of the EDIFACT message
    """
    _logger.info("Summarizing EDIFACT message via MCP tool")
    client: TransformerBeeClient = ctx.request_context.lifespan_context.transformerbeeclient
    if not edifact_format_version:
        edifact_format_version = get_current_edifact_format_version()
    try:
        # Convert EDIFACT to BO4E first
        marktnachrichten = await client.convert_to_bo4e(edifact=edifact, edifact_format_version=edifact_format_version)
        bo4e_json = "[" + ",".join(m.model_dump_json() for m in marktnachrichten) + "]"
        await ctx.info(f"Converted {len(marktnachrichten)} Marktnachricht(en) to BO4E")

        # Summarize the BO4E
        summary = await summarize_bo4e(bo4e_json)
        await ctx.info("Successfully generated summary")
        return summary
    except Exception as e:
        _logger.exception("Error while summarizing EDIFACT")
        await ctx.info(f"Error generating summary: {e}")
        raise


def main() -> None:
    """entry point for the CLI tools defined in pyproject.toml"""
    mcp.run()


if __name__ == "__main__":
    # called by 'mcp install server.py' and 'mcp dev server.py'
    main()
