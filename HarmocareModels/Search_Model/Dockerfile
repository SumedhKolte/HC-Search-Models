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

# ✅ Make sure uvicorn is installed
RUN pip install uvicorn[standard]

# Create a virtual environment (optional, for local dev)
RUN python -m venv venv
# Activate virtual environment (locally on the container)
RUN . ./venv/bin/activate && pip install -r requirements.txt

# Preload the SentenceTransformer model
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('/app/local_model')"

# Expose the port
EXPOSE 8000

# ✅ Run with python -m uvicorn
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
