# Use official Python slim image
FROM python:3.10-slim

# Set environment variables for cleaner logs and to avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install OS dependencies (needed for packages like numpy, scipy, faiss etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy all files to container
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r req/base.txt
RUN pip install -r req/prod.txt
RUN pip install -r req/dev.txt
RUN pip install -r requirements.txt

# Install additional dependencies if needed
RUN pip install faiss-cpu

# Expose the port uvicorn will run on
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
