FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    gfortran \
    git \
    libgomp1 \
    libigraph-dev \
    liblapack-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[dev]" \
    && python -m pip install notebook

RUN useradd --create-home --uid 1000 brainuser \
    && chown -R brainuser:brainuser /app

USER brainuser

EXPOSE 8888

CMD ["/bin/bash"]
