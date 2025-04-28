# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r req/base.txt
RUN pip install -r req/prod.txt
RUN pip install -r req/dev.txt
RUN pip install -r requirements.txt
RUN pip install faiss-cpu
RUN pip install "uvicorn[standard]"

# Preload the SentenceTransformer model
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('/app/local_model')"

# Expose the port
EXPOSE 8000

# Run with uvicorn directly (with --reload for dev)
CMD ["uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
