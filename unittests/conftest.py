# Testing pattern from https://fastmcp.wiki/en/patterns/testing
import pytest
from efoli import EdifactFormatVersion
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from transformerbeeclient import BOneyComb, Marktnachricht, TransformerBeeClient
from typing_extensions import AsyncGenerator, Literal

import transformerbeemcp.server as _transformerbeeservermodule
from transformerbeemcp import mcp


@pytest.fixture
def anyio_backend() -> Literal["asyncio"]:
    return "asyncio"


# pylint:disable=unused-argument
class DummyClient(TransformerBeeClient):
    """
    We mock the client because the transformer.bee client alone is already integration tested against the real backend.
    That's the big benefit of having client libraries and proper encapsulation instead of HTTP calls everywhere.
    """

    def __init__(self, host: str) -> None:
        self.host = host
        TransformerBeeClient.__init__(self)

    async def convert_to_edifact(self, boney_comb: BOneyComb, edifact_format_version: EdifactFormatVersion) -> str:
        return "dummy_edifact_message"

    async def convert_to_bo4e(self, edifact: str, edifact_format_version: EdifactFormatVersion) -> list[Marktnachricht]:
        """convert the given edifact to a list of marktnachrichten"""
        return [
            Marktnachricht(
                stammdaten=[],
                transaktionen=[BOneyComb(stammdaten=[], transaktionsdaten={"foo": "bar"})],
                nachrichtendaten={},
                UNH="dummy_unh",
            )
        ]


@pytest.fixture
def inject_dummy_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_create_client(host: str, client_id: str | None, client_secret: str | None) -> TransformerBeeClient:
        return DummyClient(host)

    # pylint:disable=protected-access
    monkeypatch.setattr(_transformerbeeservermodule, "create_client", fake_create_client)


@pytest.fixture
def transformerbee_mcp_server() -> FastMCP:
    server = mcp
    return server


@pytest.fixture
async def mcp_client(
    transformerbee_mcp_server: FastMCP, monkeypatch: pytest.MonkeyPatch, inject_dummy_client: None
) -> AsyncGenerator[Client[FastMCPTransport], None]:
    monkeypatch.setenv("TRANSFORMERBEE_HOST", "https://mock.com")
    async with Client(transport=transformerbee_mcp_server) as client:
        yield client
