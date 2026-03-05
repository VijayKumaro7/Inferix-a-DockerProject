# ── Stage 1: Train Models ─────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /app

RUN pip install --no-cache-dir scikit-learn numpy

COPY models/train.py /app/train.py
RUN python train.py

# ── Stage 2: Production Server ────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained models from trainer stage
COPY --from=trainer /app/models /app/models

# Copy app files
COPY app/main.py /app/main.py
COPY templates/ /app/templates/

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the app
CMD ["python", "main.py"]
