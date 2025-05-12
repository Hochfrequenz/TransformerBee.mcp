# mainly copied from here:
# https://github.com/modelcontextprotocol/python-sdk/blob/babb477dffa33f46cdc886bc885eb1d521151430/tests/shared/test_memory.py#L1-L48
import pytest
from efoli import EdifactFormatVersion
from mcp.client.session import ClientSession
from mcp.server import FastMCP
from mcp.shared.memory import (
    create_connected_server_and_client_session,
)
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
async def client_connected_to_server(
    transformerbee_mcp_server: FastMCP, monkeypatch: pytest.MonkeyPatch, inject_dummy_client: None
) -> AsyncGenerator[ClientSession, None]:
    monkeypatch.setenv("TRANSFORMERBEE_HOST", "https://mock.com")
    # pylint:disable=protected-access
    async with create_connected_server_and_client_session(transformerbee_mcp_server._mcp_server) as client_session:
        yield client_session
