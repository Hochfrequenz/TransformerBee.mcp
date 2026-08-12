FROM python:3.14-slim
LABEL authors="Hochfrequenz Unternehmensberatung GmbH"

# Pull the uv binary into the image (the base image ships pip but no uv).
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # use the interpreter that ships with the base image instead of downloading a managed CPython
    UV_PYTHON_DOWNLOADS=never

RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Install the (runtime) dependencies from the lockfile without installing the project itself.
# `--no-install-project` preserves exactly the previous `pip install -r requirements.txt` behaviour:
# `pip install .` / installing the project fails in this image because the hatch-vcs version is
# undefined without a git checkout:
# LookupError: Error getting the version from source `vcs`: setuptools-scm was unable to detect version for /app
# That's why we cannot use the `uv run fastmcp ...` CLI shortcut in the entrypoint below either.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# put the virtual environment created by `uv sync` on the PATH
ENV PATH="/app/.venv/bin:$PATH"

USER appuser

# the tail command is to not directly exit after starting the server
# feel free to remove it, but please manually test your changes ;)
ENTRYPOINT ["sh", "-c", "fastmcp run src/transformerbeemcp/server.py && tail -f /dev/null"]
