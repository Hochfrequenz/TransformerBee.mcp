# Dockerfile for REST API (default)
# For MCP server, use Dockerfile.mcp instead
# Tested: Docker build succeeds with Python 3.14-slim

FROM python:3.14-slim
LABEL authors="Hochfrequenz Unternehmensberatung GmbH"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Expose port for REST API
EXPOSE 8080

# Use FastAPI CLI for production (official approach)
CMD ["fastapi", "run", "src/transformerbeemcp/rest_api.py", "--port", "8080", "--host", "0.0.0.0"]
