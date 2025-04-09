FROM python:3.10-slim

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy only requirements first (layer caching)
COPY requirements/base.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install FAISS
RUN pip install faiss-cpu

# Copy the full codebase
COPY . /app

# Run server (replace if different)
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
