# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Set environment paths
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Copy dependency files first for better caching
COPY requirements.txt .
COPY lightrag/api/requirements.txt ./lightrag/api/
COPY pyproject.toml .
COPY uv.lock* ./

# Install dependencies using uv for faster builds
RUN uv pip install --system --no-cache-dir -r requirements.txt
RUN uv pip install --system --no-cache-dir -r lightrag/api/requirements.txt

# Install dependencies for default storage
RUN uv pip install --system --no-cache-dir nano-vectordb networkx
# Install dependencies for default LLM
RUN uv pip install --system --no-cache-dir openai ollama tiktoken
# Install dependencies for default document loader
RUN uv pip install --system --no-cache-dir pypdf2 python-docx python-pptx openpyxl

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies and uv
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ./lightrag ./lightrag
COPY setup.py .
COPY pyproject.toml .
COPY .env .env
# Install the application
RUN pip install ".[api]"

# Create necessary directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs

# Environment variables
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV PATH="/root/.local/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9621/health || exit 1

# Expose the default port
EXPOSE 9621

# Set entrypoint to multiuser server
ENTRYPOINT ["python", "-m", "lightrag.api.lightrag_multiuser_server"]
