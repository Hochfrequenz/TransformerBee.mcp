FROM python:3.13-slim
LABEL authors="Hochfrequenz Unternehmensberatung GmbH"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# If you try to run `pip install .` with a pyproject.toml, you'll have problems with the build, because the git tag version is undefined.
# LookupError: Error getting the version from source `vcs`: setuptools-scm was unable to detect version for /app
# That's why we cannot use the CLI shortcut in the entrypoint below.

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

USER appuser

ENTRYPOINT ["sh", "-c", "mcp run src/transformerbeemcp/server.py && tail -f /dev/null"]
