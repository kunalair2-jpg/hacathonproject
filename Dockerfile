FROM python:3.11-slim

WORKDIR /app

# Install Python deps first (layer cache friendly)
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "fastapi>=0.110" \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    "openenv-core" \
    "openai" \
    "numpy" \
    "pillow" \
    "requests"

# Copy source
COPY . .

ENV PYTHONUNBUFFERED=1
ENV TASK_NAME=hard
ENV PORT=8000

EXPOSE 8000

CMD ["python", "-m", "server.app"]
