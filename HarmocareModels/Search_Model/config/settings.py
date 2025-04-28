from pathlib import Path
import os
from datetime import datetime, timezone
from typing import Dict, Any
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / 'ml'
SSL_DIR = PROJECT_ROOT / 'ssl'

# Create necessary directories
ML_DIR.mkdir(parents=True, exist_ok=True)
SSL_DIR.mkdir(parents=True, exist_ok=True)

# Create required directories
for dir_path in [
    ML_DIR / "models",
    ML_DIR / "indexes",
    ML_DIR / "embeddings",
    ML_DIR / "data",
    SSL_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)

def download_ssl_cert() -> str:
    """Download and setup SSL certificate"""
    cert_path = SSL_DIR / 'rds-ca-bundle.pem'
    cert_timestamp_path = SSL_DIR / 'cert_timestamp.txt'
    
    needs_update = True
    if cert_path.exists() and cert_timestamp_path.exists():
        with open(cert_timestamp_path, 'r') as f:
            last_update = datetime.fromisoformat(f.read().strip())
            if (datetime.now(timezone.utc) - last_update).days < 30:
                needs_update = False
    
    if needs_update:
        try:
            cert_url = 'https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem'
            response = requests.get(cert_url)
            response.raise_for_status()
            
            with open(cert_path, 'wb') as f:
                f.write(response.content)
            
            timestamp = datetime.now(timezone.utc)
            with open(cert_timestamp_path, 'w') as f:
                f.write(timestamp.isoformat())
        except requests.RequestException as e:
            print(f"Failed to download SSL certificate: {e}")
            raise
    
    return str(cert_path)

# SSL Certificate setup
SSL_CERT_PATH = download_ssl_cert()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'medical_search'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': os.getenv('DB_PORT', '5432')
}

# Entity configurations
ENTITIES = {
    'doctors': {
        'table': 'doctors',
        'id_field': 'did',
        'embedding_field': 'embedding',
        'search_vector': 'search_vector',
        'searchable_fields': [
            'name', 'specialization', 'city', 'degree',
            'experience', 'rating'
        ],
        'filterable_fields': {
            'gender': str,
            'city': str,
            'specialization': str,
            'rating': float,
            'experience': int,
            'consultantfee': int
        }
    },
    'hospitals': {
        'table': 'hospitals',
        'id_field': 'hid',
        'embedding_field': 'embedding',
        'search_vector': 'search_vector',
        'searchable_fields': ['name', 'hospital_type', 'location'],
        'filterable_fields': {
            'hospital_type': str,
            'rating': float,
            'location': str
        }
    },
    # ... similar configurations for other entities
}

# Search configuration
SEARCH_CONFIG = {
    'default_limit': 10,
    'max_limit': 100,
    'min_score': 0.5,
    'cache_ttl': 3600,  # 1 hour
    'timeout': 30  # seconds
}

# Model configuration
MODEL_CONFIG = {
    'base_model': os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'),
    'embedding_dim': 384,  # Fixed for all-MiniLM-L6-v2
    'batch_size': int(os.getenv('BATCH_SIZE', '32')),
    'training': {
        'epochs': 3,
        'evaluation_steps': 100,
        'warmup_steps': 100
    }
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'workers': 1,
    'log_level': 'info'
}

# SSL Certificate configuration
SSL_CONFIG = {
    'cert_file': SSL_DIR / 'rds-ca-bundle.pem',
    'timestamp_file': SSL_DIR / 'cert_timestamp.txt',
    'update_interval_days': 30,
    'aws_cert_url': 'https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem'
}