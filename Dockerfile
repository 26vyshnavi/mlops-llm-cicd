# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Multi-stage build for the LLM Summarizer app
#
# WHY MULTI-STAGE?
#   Stage 1 (builder): Install all Python packages, including heavy build deps
#                       (compilers, headers) that are needed at install time
#                       but NOT at runtime.
#   Stage 2 (runtime): Copy only the installed packages and source code.
#                       No build tools, smaller final image.
#
#   Typical result: ~1.5 GB (with PyTorch CPU) vs. ~2.5 GB single-stage.
#
# WHY python:3.11-slim?
#   Slim variants strip documentation, test suites, and package manager caches.
#   They're based on Debian and have apt-get available (unlike alpine, which
#   causes glibc/musl issues with PyTorch binaries).
#
# HUGGING FACE SPACES REQUIREMENTS:
#   - Listen on port 7860 (HF proxies this port)
#   - Run as a non-root user (HF enforces this for Docker Spaces)
#   - EXPOSE 7860 in the Dockerfile
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
# git: sometimes needed by pip to install from GitHub sources
# curl: used in the health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — this layer is cached by Docker unless
# requirements.txt changes, making rebuilds much faster on code-only changes.
COPY requirements.txt .

# Install into a prefix directory so we can copy cleanly to Stage 2.
# --no-cache-dir reduces image size (pip's HTTP cache is not needed after install).
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code
# We copy src/ and app/ separately so that changes to app/ don't
# invalidate the src/ layer cache (and vice versa).
COPY src/ ./src/
COPY app/ ./app/

# ── Security: run as non-root ─────────────────────────────────────────────────
# Running as root inside a container is a security risk — if an attacker escapes
# the container, they have root on the host. This creates a minimal user.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Metadata ─────────────────────────────────────────────────────────────────
# LABEL provides provenance info — visible in Docker Hub and image inspect.
LABEL org.opencontainers.image.title="DistilGPT-2 Summarizer"
LABEL org.opencontainers.image.description="Fine-tuned DistilGPT-2 for text summarization"
LABEL org.opencontainers.image.source="https://github.com/YOUR_USERNAME/mlops-llm-cicd"

# HuggingFace Spaces (and most reverse proxies) expect port 7860
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
# Docker and orchestrators (Kubernetes, ECS) use this to decide if the
# container is healthy. Gradio exposes /  as its root — a 200 response = healthy.
# --start-period=120s: allow 2 minutes for model loading before checks begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail --silent http://localhost:7860/ || exit 1

# ── Entrypoint ───────────────────────────────────────────────────────────────
# CMD (not ENTRYPOINT) so you can override it with:
#   docker run ... python -m pytest   ← for debugging
CMD ["python", "app/app.py"]
