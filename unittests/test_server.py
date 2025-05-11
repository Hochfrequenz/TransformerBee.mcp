import pytest
from mcp import ClientSession
from mcp.server import FastMCP
from mcp.types import EmptyResult


@pytest.mark.anyio
async def test_fastmcp_server_initialization(transformerbee_mcp_server:FastMCP)->None:
    """Testet, ob der FastMCP-Server korrekt initialisiert wird."""
    assert transformerbee_mcp_server is not None
    assert transformerbee_mcp_server.name == "TransformerBee.mcp"
    assert "transformerbeeclient" in transformerbee_mcp_server.dependencies


@pytest.mark.anyio
async def test_convert_edifact_to_bo4e(client_connected_to_server:FastMCP)->None:
    """Testet das Tool zum Konvertieren von EDIFACT zu BO4E."""
    edifact_message = "UNA:+.? 'UNB+UNOC:3+1234567890123:14+9876543210987:14+210101:1234+00000000000778++ORDERS'"

    response = await client_connected_to_server.call_tool(
        "convert_edifact_to_bo4e",
        {"edifact": edifact_message}
    )
    assert isinstance(response, dict)
    assert "transaktionen" in response

@pytest.mark.anyio
async def test_memory_server_and_client_connection(
    client_connected_to_server: ClientSession,
)->None:
    """Shows how a client and server can communicate over memory streams."""
    response = await client_connected_to_server.send_ping()
    assert isinstance(response, EmptyResult)