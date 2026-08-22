FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY app ./app

RUN python -m pip install --upgrade pip && \
    python -m pip install ".[demo,retrieval]"

EXPOSE 8000

CMD ["shopee-demo", "api", "--config", "configs/serving/demo.yaml", "--host", "0.0.0.0", "--port", "8000"]
