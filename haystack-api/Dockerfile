FROM python:3.11-slim

WORKDIR /app

# M3 video pipeline system deps (never added when the Python packages were
# first written, so every M3 phase past download has been undeployable
# until now — see project-knowledge-base/02-current-system-state.md).
# ffmpeg: extract_audio() shells out to it directly. libglib2.0-0: the
# common missing-symbol error for opencv-python-headless on a slim image
# even though it has no GUI/libGL dependency itself.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    rclone \
    && rm -rf /var/lib/apt/lists/*

# No build-essential: every Phase 1 dependency ships a manylinux wheel.
# Restore it if a future dep needs to compile from source.

# Build context is the repo root (compose.yml), not this directory, so every
# path below is prefixed haystack-api/ — needed to also reach
# processing/telemetry/ (D52).

# Install Python dependencies
COPY haystack-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY haystack-api/ .

# Shared observability module (rag-AI/CLAUDE.md "Telemetry module") — the one
# place that imports Phoenix/OpenTelemetry, reused via telemetry_compat.py
# rather than duplicated, so a backend swap stays a one-file change.
COPY processing/telemetry/ ./telemetry/

# Create upload directory
RUN mkdir -p /data/haystack-uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
