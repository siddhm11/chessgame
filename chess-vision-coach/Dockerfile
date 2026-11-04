FROM python:3.11-slim

# System deps: OpenCV headless needs libGL, Pillow/CairoSVG need Cairo + fonts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf2.0-0 \
        libffi-dev \
        stockfish \
    && rm -rf /var/lib/apt/lists/*

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
