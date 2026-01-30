# Use official Python runtime as base image
FROM python:3.10-slim-bullseye

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    cmake \
    libigraph-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -u 1000 brainuser && \
    chown -R brainuser:brainuser /app
USER brainuser

# Expose Jupyter port (optional)
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["/bin/bash"]

# For Jupyter notebook usage, uncomment:
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
