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

# Copy only the files needed for inference
COPY requirements.txt .
COPY req/base.txt ./req/base.txt
COPY req/prod.txt ./req/prod.txt
COPY app.py .
COPY local_model ./local_model

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r req/base.txt
RUN pip install -r req/prod.txt
RUN pip install -r requirements.txt
RUN pip install gradio

# Expose the port Gradio will run on
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app.py"]
