FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl gosu && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY main.py .
COPY bot/ ./bot/

# Create non-root user for security
RUN groupadd -r bot 2>/dev/null || true && useradd -r -g bot -d /app -s /sbin/nologin bot 2>/dev/null || true \
    && chown -R bot:bot /app \
    && mkdir -p /app/vector_db \
    && chown -R bot:bot /app/vector_db

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:9101/health || exit 1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Entrypoint runs as root to fix volume permissions, then drops to bot user via gosu.
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py"]
