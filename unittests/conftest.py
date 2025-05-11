# mainly copied from here:
# https://github.com/modelcontextprotocol/python-sdk/blob/babb477dffa33f46cdc886bc885eb1d521151430/tests/shared/test_memory.py#L1-L48
import pytest
from mcp.client.session import ClientSession
from mcp.server import Server
from mcp.shared.memory import (
    create_connected_server_and_client_session,
)
from mcp.types import (
    Resource,
)
from pydantic import AnyUrl
from typing_extensions import AsyncGenerator, Literal

from transformerbeemcp import mcp


@pytest.fixture
def anyio_backend()->Literal["asyncio"]:
    return "asyncio"


@pytest.fixture
def transformerbee_mcp_server() -> Server:
    server = mcp
    return server


@pytest.fixture
async def client_connected_to_server(
    transformerbee_mcp_server: Server,
) -> AsyncGenerator[ClientSession, None]:
    async with create_connected_server_and_client_session(transformerbee_mcp_server) as client_session:
        yield client_session