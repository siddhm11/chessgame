# Pinned by digest: an unpinned python:3.11-slim moved to Debian 13 (trixie),
# where one of the old apt packages no longer exists and the build failed.
# This digest is python:3.11-slim-bookworm (Debian 12).
FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd

# System packages, and only these: Stockfish, and Cairo (CairoSVG renders
# board images). The Python wheels (OpenCV, onnxruntime, Pillow, numpy) bundle
# their own native libraries -- checked with ldd: they need only libc,
# libstdc++, libgcc and zlib, which the base image already has.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcairo2 \
        stockfish \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces runs the container as a non-root user (UID 1000).
# Create that user and a writable HOME so Ultralytics / matplotlib /
# fontconfig caches don't try to write to /root.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app
RUN chown -R user:user /app
USER user

# Install Python deps before copying source so layer is cached on dep changes.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application source.
COPY --chown=user:user . .

# Uploads directory must survive container restarts; mount a volume here.
RUN mkdir -p /app/uploads

EXPOSE 8000

# CVC_BACKEND defaults to model (YOLO fine-tuned).
# Override with -e CVC_BACKEND=classical for diagram-only use.
ENV CVC_BACKEND=model

# Cap glibc malloc arenas. Threadpool requests each allocate ~8 MB image
# buffers; with per-thread arenas RSS crept 350 -> 479 MB over a soak test,
# with 2 arenas it stayed flat at ~315 MB (and p95 latency improved).
ENV MALLOC_ARENA_MAX=2

# Single worker: session store is in-process memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
