# TransformerBee.MCP

This is a Model Context Protocol (MCP) server and REST API for [transformer.bee](https://github.com/enercity/edifact-bo4e-converter/), written in Python.
Under the hood it uses [`python-mcp`](https://github.com/modelcontextprotocol/python-sdk) and [`transformerbeeclient.py`](https://github.com/Hochfrequenz/TransformerBeeClient.py).

## Features

- **MCP Server**: Expose transformer.bee conversion tools to AI assistants (Claude Desktop, etc.)
- **REST API**: HTTP endpoint for EDIFACT summarization using a local LLM (Ollama)
- **EDIFACT Summarization**: Generate human-readable German summaries of EDIFACT messages

## Environment Variables

All environment variables used by this package:

### MCP Server & Transformer.bee Client

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `TRANSFORMERBEE_HOST` | URL of the transformer.bee backend | - | **Yes** |
| `TRANSFORMERBEE_CLIENT_ID` | OAuth client ID for authenticated requests | - | No |
| `TRANSFORMERBEE_CLIENT_SECRET` | OAuth client secret | - | No |

### REST API (Summarization)

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OLLAMA_HOST` | URL of the Ollama instance | `http://localhost:11434` | No |
| `OLLAMA_MODEL` | LLM model to use for summarization | `llama3` | No |
| `AUTH0_DOMAIN` | Auth0 domain for JWT verification | `hochfrequenz.eu.auth0.com` | No |
| `AUTH0_AUDIENCE` | Auth0 API audience identifier | `https://transformer.bee` | No |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | `http://localhost:5173,...` | No |
| `RATE_LIMIT` | Max requests per user per window | `10` | No |
| `RATE_WINDOW_SECONDS` | Rate limit window duration | `60` | No |
| `PORT` | REST API server port | `8080` | No |
| `HOST` | REST API server host | `0.0.0.0` | No |

## Installation

You can install the MCP server as Python package or pull the Docker image.

### Install as Python Package
```shell
uv install transformerbeemcp
```
or if you are using `pip`:
```sh
pip install transformerbeemcp
```

### Install as Docker Image

There are two Dockerfiles for different use cases:

| Dockerfile | Purpose | Use Case |
|------------|---------|----------|
| `Dockerfile` | REST API server (FastAPI) | Web apps calling `/summarize` endpoint |
| `Dockerfile.mcp` | MCP server (stdio) | AI assistants like Claude Desktop |

**Build locally:**
```sh
# Build REST API image
docker build -t transformerbee-rest -f Dockerfile .

# Build MCP Server image
docker build -t transformerbee-mcp -f Dockerfile.mcp .
```

**Or pull from GHCR:**
```sh
# REST API (default)
docker pull ghcr.io/hochfrequenz/transformerbee.mcp:latest

# MCP Server
docker pull ghcr.io/hochfrequenz/transformerbee.mcp-mcp:latest
```

## MCP Server

### Start via CLI
In a terminal **inside the virtual environment** where you installed the package:

```sh
(myvenv) run-transformerbee-mcp-server
```

### Start via Docker
```sh
docker run --network host -i --rm \
  -e TRANSFORMERBEE_HOST=http://localhost:5021 \
  ghcr.io/hochfrequenz/transformerbee.mcp:latest
```

### Register in Claude Desktop

#### If you checked out this repository
```sh
cd path/to/reporoot/src/transformerbeemcp
mcp install server.py
```

#### If you installed the package via pip/uv
Modify your `claude_desktop_config.json` (found in Claude Desktop menu via "Datei > Einstellungen > Entwickler > Konfiguration bearbeiten"):
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
where `C:\github\MyProject\.myvenv` is the path to your virtual environment and `localhost:5021` exposes transformer.bee running in a docker container.

Note that this package marks `uv` as a dev-dependency, so you might need to install it `pip install transformerbeempc[dev]` as a lot of MCP tooling assumes you have `uv` installed.

For details about the environment variables and/or starting transformer.bee locally, check [`transformerbeeclient.py`](https://github.com/Hochfrequenz/TransformerBeeClient.py) docs.

#### If you installed the package via Docker
```json
{
  "mcpServers": {
    "TransformerBee.mcp": {
      "command": "docker",
      "args": [
        "run",
        "--network",
        "host",
        "-i",
        "--rm",
        "-e",
        "TRANSFORMERBEE_HOST=http://localhost:5021",
        "ghcr.io/hochfrequenz/transformerbee.mcp:latest"
      ],
      "env": {
        "TRANSFORMERBEE_HOST": "http://localhost:5021",
        "TRANSFORMERBEE_CLIENT_ID": "",
        "TRANSFORMERBEE_CLIENT_SECRET": ""
      }
    }
  }
}
```
I'm aware that using the `--network host` option is a bit hacky and not best practice.

## REST API for EDIFACT Summarization

The package also includes a REST API that uses a local LLM (Ollama) to generate human-readable German summaries of EDIFACT messages.

### Start the REST API Server

#### Via CLI
```sh
(myvenv) run-transformerbee-rest-api
```

#### Via Docker
```sh
docker run -p 8080:8080 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e AUTH0_DOMAIN=hochfrequenz.eu.auth0.com \
  -e AUTH0_AUDIENCE=https://transformer.bee \
  transformerbee-rest
```

#### Via Docker Compose (with Ollama sidecar)

The `docker-compose.yml` provides a complete setup for running the summarizer with a local Ollama instance:

```
┌─────────────────────────────────────────────────────────────┐
│                    docker-compose                            │
│  ┌─────────────────┐         ┌─────────────────────────┐   │
│  │   summarizer    │────────▶│        ollama           │   │
│  │   (REST API)    │         │   (Llama 3 model)       │   │
│  │   Port 8080     │         │   Port 11434            │   │
│  └─────────────────┘         └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Services:**
- `summarizer`: The REST API container (from `Dockerfile`)
- `ollama`: Local LLM server running Llama 3
- `ollama-init`: One-time init container to pull the model

**Usage:**
```sh
# Start the services
docker-compose up -d

# Pull the LLM model (first time only)
docker-compose run --rm --profile init ollama-init

# Check logs
docker-compose logs -f summarizer

# Stop services
docker-compose down
```

The summarizer connects to Ollama via the internal Docker network (`http://ollama:11434`).

### API Endpoints

#### `POST /summarize`
Generate a German summary of an EDIFACT message.

**Request:**
```json
{
  "edifact": "UNB+UNOC:3+9904321000019:500+9900123000003:500+241210:1245+ABC123456789++TL'"
}
```

**Response:**
```json
{
  "summary": "Dies ist eine Zählerstandsmeldung vom Lieferanten (GLN: 9904321000019) an den Netzbetreiber..."
}
```

**Authentication:** Requires a valid Auth0 bearer token (same audience as transformer.bee).

**Rate Limiting:** 10 requests per minute per user (configurable via `RATE_LIMIT` env var).

#### `GET /health`
Health check endpoint (no authentication required).

**Response:**
```json
{
  "status": "ok"
}
```

## Deployment Guide

### Quickstart: Run as Sidecar

The fastest way to run the summarization service alongside your existing setup:

```sh
# Start Ollama + pull model (one-time)
docker run -d --name ollama -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama
docker exec ollama ollama pull llama3

# Start the summarizer
docker run -d --name summarizer \
  -p 8080:8080 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ghcr.io/hochfrequenz/transformerbee.mcp:latest
```

### Production Deployment

If you control the deployment of **marktnachrichten-dolmetscher** and/or **transformer.bee**:

```
┌───────────────────────────────────────────────────────────────────────┐
│                        Your Infrastructure                             │
│                                                                        │
│  ┌──────────────────────┐                                             │
│  │ marktnachrichten-    │                                             │
│  │ dolmetscher          │                                             │
│  │ (Frontend)           │                                             │
│  └─────────┬────────────┘                                             │
│            │                                                           │
│            ├─────────────────────┬──────────────────────┐             │
│            ▼                     ▼                      ▼             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │ transformer.bee  │  │   summarizer     │  │     ollama       │    │
│  │ (EDIFACT↔BO4E)   │  │   (REST API)     │──│   (Llama 3)      │    │
│  │ Port: 5021       │  │   Port: 8080     │  │   Port: 11434    │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
```

**Recommended setup:**
1. Deploy `ollama` with a persistent volume for model storage
2. Deploy `summarizer` (this package's REST API) pointing to Ollama
3. Configure `marktnachrichten-dolmetscher` to call both:
   - `transformer.bee` for EDIFACT↔BO4E conversion
   - `summarizer` for human-readable summaries

**Auth0 integration:** The summarizer uses the same Auth0 audience (`https://transformer.bee`) as transformer.bee.
Clients already authenticated with transformer.bee can reuse their tokens—no additional Auth0 configuration needed.

**Environment variables for production:**
```sh
# Required for summarizer
OLLAMA_HOST=http://ollama:11434  # Internal Docker network
OLLAMA_MODEL=llama3

# Auth (defaults work if using same Auth0 tenant as transformer.bee)
AUTH0_DOMAIN=hochfrequenz.eu.auth0.com
AUTH0_AUDIENCE=https://transformer.bee

# CORS (add your production frontend URLs)
ALLOWED_ORIGINS=https://your-dolmetscher.example.com

# Rate limiting (adjust as needed)
RATE_LIMIT=10
RATE_WINDOW_SECONDS=60
```
