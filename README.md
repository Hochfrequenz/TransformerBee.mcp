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
where `C:\github\MyProject\.myvenv` is the path to your virtual environment where you installed the package and `localhost:5021` exposes transformer.bee running in a docker container.
Alternatively, if you haven't configured this handy CLI command
https://github.com/Hochfrequenz/TransformerBee.mcp/blob/c0898769670469df13f23b57a55fe4b71ed9795b/pyproject.toml#L101-L102

you can just call python with non-empty args.

Note that this package marks `uv` as a dev-dependency, so you might need to install it `pip install transformerbeempc[dev]` in your virtual environment as well as a lot of MCP tooling assumes you have `uv` installed.

For details about the environment variables and/or starting transformer.bee locally, check [`transformerbeeclient.py`](https://github.com/Hochfrequenz/TransformerBeeClient.py) docs.
