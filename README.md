# TransformerBee.MCP

This is a simple PoC of a Model Context Protocol (MCP) server for [transformer.bee](https://github.com/enercity/edifact-bo4e-converter/), written in Python.
Under the hood it uses [`python-mdc`](https://github.com/modelcontextprotocol/python-sdk) and [`transformerbeeclient.py`](https://github.com/Hochfrequenz/TransformerBeeClient.py).

## Installation
```shell
uv install transformerbeemcp
```
or if you are using `pip`:
```sh
pip install transformerbeemcp
```

## Start the Server inside the CLI
The package ships a simple CLI argument to start the server.
In a terminal **inside the virtual environment in which you installed the package (here `myvenv`)**, call:

```sh
(myvenv) run-transformerbee-mcp-server
```

## Install directly into Claude Desktop
### If you checked out this repository
```sh
cd path/to/reporoot/src/transformerbeemcp
mcp install server.py
```
### If you installed the package via pip/uv
Modify your `claude_desktop_config.json` (that can be found in Claude Desktop menu via "Datei > Einstellungen > Entwickler > Konfiguration bearbeiten"):
```json
{
  "mcpServers": {
    "TransformerBee.mcp": {
      "command": "C:\\github\\MyProject\\.myvenv\\Scripts\\run-transformerbee-mcp-server.exe",
      "args": [],
      "env": {
        "TRANSFORMERBEE_HOST": "http://localhost:5021",
        "TRANSFORMERBEE_CLIENT_ID": "",
        "TRANSFORMERBEE_CLIENT_SECRET": ""
      }
    }
  }
}
```
where `C:\path\to\myvenv` is the path to your virtual environment where you installed the package and `localhost:5021` exposes transformer.bee running in a docker container.

For details about the environment variables and/or starting transformer.bee locally, check [`transformerbeeclient.py`](https://github.com/Hochfrequenz/TransformerBeeClient.py) docs.
