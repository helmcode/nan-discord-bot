FROM python:3.11-slim

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
  CMD kill -0 1 || exit 1

USER bot

CMD ["python", "main.py"]
