import pytest
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport
from transformerbeeclient import BOneyComb


@pytest.mark.anyio
async def test_fastmcp_server_initialization(transformerbee_mcp_server: FastMCP) -> None:
    """Testet, ob der FastMCP-Server korrekt initialisiert wird."""
    assert transformerbee_mcp_server is not None
    assert transformerbee_mcp_server.name == "TransformerBee.mcp"


@pytest.mark.anyio
async def test_list_tools(mcp_client: Client[FastMCPTransport]) -> None:
    """Testet, ob die erwarteten Tools registriert sind."""
    tools = await mcp_client.list_tools()
    tool_names = {tool.name for tool in tools}
    assert "convert_edifact_to_bo4e" in tool_names
    assert "convert_bo4e_to_edifact" in tool_names


@pytest.mark.anyio
async def test_convert_edifact_to_bo4e(mcp_client: Client[FastMCPTransport]) -> None:
    """Testet das Tool zum Konvertieren von EDIFACT zu BO4E."""
    edifact_message = "UNA:+.? 'UNB+UNOC:3+1234567890123:14+9876543210987:14+210101:1234+00000000000778++ORDERS'"

    result = await mcp_client.call_tool("convert_edifact_to_bo4e", {"edifact": edifact_message})
    assert result.is_error is False
    assert result.data is not None
    assert result.data.stammdaten == []
    assert result.data.transaktionsdaten["foo"] == "bar"


@pytest.mark.anyio
async def test_convert_bo4e_to_edifact(mcp_client: Client[FastMCPTransport]) -> None:
    """Testet das Tool zum Konvertieren von BO4E zu EDIFACT."""
    result = await mcp_client.call_tool(
        "convert_bo4e_to_edifact",
        {"transaktion": BOneyComb(transaktionsdaten={}, stammdaten=[]), "edifact_format_version": "FV2504"},
    )
    assert result.is_error is False
    assert result.data == "dummy_edifact_message"


@pytest.mark.anyio
async def test_client_connection(mcp_client: Client[FastMCPTransport]) -> None:
    """Testet, ob der Client korrekt mit dem Server verbunden ist."""
    assert mcp_client.is_connected()
    await mcp_client.ping()
