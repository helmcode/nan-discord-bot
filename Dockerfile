FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY main.py .
COPY bot/ ./bot/

CMD ["python", "main.py"]
