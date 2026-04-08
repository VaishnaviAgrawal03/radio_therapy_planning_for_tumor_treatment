# ─────────────────────────────────────────────────────────────────────────────
# RadiotherapyPlanningEnv — Docker Image
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build: keeps final image lean (~800MB vs ~3GB)
#
# Build:
#   docker build -t radiotherapy-env:latest .
#
# Run environment (interactive Python):
#   docker run -it radiotherapy-env:latest python
#
# Run Gradio demo:
#   docker run -p 7860:7860 radiotherapy-env:latest python app/app.py
#
# Run training:
#   docker run radiotherapy-env:latest python baseline/train_ppo.py --task prostate
#
# Run tests:
#   docker run radiotherapy-env:latest pytest tests/ -v
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLBACKEND=Agg

# ── Expose Gradio port ────────────────────────────────────────────────────────
EXPOSE 7860

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import radiotherapy_env; import gymnasium as gym; \
    env = gym.make('RadiotherapyEnv-prostate-v1'); env.reset(); env.close(); print('OK')"

# ── Default command: run FastAPI server ──────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
