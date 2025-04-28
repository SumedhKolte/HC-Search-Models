from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv
import requests
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def download_ssl_cert() -> str:
    """Download and setup SSL certificate from AWS"""
    timestamp = datetime.now(timezone.utc)
    ssl_dir = Path(__file__).parent.parent / 'ssl'
    ssl_dir.mkdir(exist_ok=True)
    
    cert_path = ssl_dir / 'rds-ca-bundle.pem'
    cert_timestamp_path = ssl_dir / 'cert_timestamp.txt'
    
    needs_update = True
    if cert_path.exists() and cert_timestamp_path.exists():
        with open(cert_timestamp_path, 'r') as f:
            last_update = datetime.fromisoformat(f.read().strip())
            # Check if cert is less than 30 days old
            if (timestamp - last_update).days < 30:
                needs_update = False
    
    if needs_update:
        try:
            cert_url = 'https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem'
            response = requests.get(cert_url)
            response.raise_for_status()
            
            with open(cert_path, 'wb') as f:
                f.write(response.content)
            
            with open(cert_timestamp_path, 'w') as f:
                f.write(timestamp.isoformat())
            
            logger.info(f"SSL certificate downloaded at {timestamp.isoformat()}")
        except Exception as e:
            logger.error(f"Failed to download SSL certificate: {str(e)}")
            raise
    
    return str(cert_path)

# Get SSL certificate path
SSL_CERT_PATH = download_ssl_cert()

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DIRS = {
    'ml': {
        'root': PROJECT_ROOT / 'ml',
        'models': PROJECT_ROOT / 'ml' / 'models',
        'indexes': PROJECT_ROOT / 'ml' / 'indexes',
        'embeddings': PROJECT_ROOT / 'ml' / 'embeddings',
        'data': PROJECT_ROOT / 'ml' / 'data'
    },
    'ssl': PROJECT_ROOT / 'ssl',
    'monitoring': {
        'metrics': PROJECT_ROOT / 'monitoring' / 'metrics',
        'logs': PROJECT_ROOT / 'monitoring' / 'logs'
    }
}

# Create necessary directories
for category in DIRS.values():
    if isinstance(category, dict):
        for dir_path in category.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    else:
        category.mkdir(parents=True, exist_ok=True)

# Database schema configuration
SCHEMA = {
    'doctors': {
        'table': 'doctors',
        'id_field': 'did',
        'vector_field': 'embedding',
        'text_search_field': 'search_vector',
        'searchable_fields': [
            'name', 'specialization', 'city', 'degree',
            'experience', 'rating', 'weighted_rating'
        ],
        'filterable_fields': [
            'gender', 'city', 'specialization', 'rating',
            'consultantfee'
        ]
    },
    'hospitals': {
        'table': 'hospitals',
        'id_field': 'hid',
        'vector_field': 'embedding',
        'text_search_field': 'search_vector',
        'searchable_fields': ['name', 'hospital_type', 'location'],
        'filterable_fields': ['hospital_type', 'rating', 'location']
    },
    'clinics': {
        'table': 'clinics',
        'id_field': 'cid',
        'vector_field': 'embedding',
        'text_search_field': 'search_vector',
        'searchable_fields': ['name', 'location'],
        'filterable_fields': ['location']
    },
    'diseases': {
        'table': 'diseases',
        'id_field': 'disease_id',
        'vector_field': 'embedding',
        'text_search_field': 'search_vector',
        'searchable_fields': ['name', 'common_names', 'description'],
        'filterable_fields': ['tags']
    },
    'symptoms': {
        'table': 'symptoms',
        'id_field': 'symptom_id',
        'vector_field': 'embedding',
        'text_search_field': 'search_vector',
        'searchable_fields': ['name', 'description'],
        'filterable_fields': ['tags']
    }
}

# Logging tables configuration
LOGGING_TABLES = {
    'search_logs': {
        'table': 'search_logs',
        'required_fields': ['query', 'search_type', 'filters']
    },
    'model_training_metrics': {
        'table': 'model_traing_metrics',
        'required_fields': ['entity_type', 'embeddings_count', 'dimension']
    },
    'query_expansion_logs': {
        'table': 'query_expansion_logs',
        'required_fields': ['original_query', 'expanded_terms']
    },
    'search_metrics': {
        'table': 'search_metrics',
        'required_fields': ['query', 'total_results', 'execution_time']
    }
}

# Database configuration with SSL
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME', 'medical_search'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'min_connections': int(os.getenv('DB_MIN_CONN', '1')),
    'max_connections': int(os.getenv('DB_MAX_CONN', '10')),
    'sslmode': 'verify-full',
    'sslcert': SSL_CERT_PATH
}

# Search configuration
SEARCH_CONFIG = {
    'default_limit': int(os.getenv('SEARCH_DEFAULT_LIMIT', '10')),
    'min_similarity_score': float(os.getenv('MIN_SIMILARITY_SCORE', '0.5')),
    'max_query_expansion_terms': int(os.getenv('MAX_EXPANSION_TERMS', '5')),
    'location_radius_km': float(os.getenv('DEFAULT_RADIUS_KM', '10.0')),
    'cache_results': os.getenv('CACHE_RESULTS', 'true').lower() == 'true'
}

# Model configuration
MODEL_CONFIG = {
    'base_model': os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2'),
    'embedding_dim': 384,  # Default for all-MiniLM-L6-v2
    'batch_size': int(os.getenv('MODEL_BATCH_SIZE', '32')),
    'max_seq_length': int(os.getenv('MAX_SEQ_LENGTH', '128')),
    'training': {
        'epochs': int(os.getenv('TRAIN_EPOCHS', '5')),
        'learning_rate': float(os.getenv('LEARNING_RATE', '2e-5')),
        'warmup_steps': int(os.getenv('WARMUP_STEPS', '100')),
        'evaluation_steps': int(os.getenv('EVAL_STEPS', '1000'))
    }
}

# Monitoring configuration
MONITORING_CONFIG = {
    'enable_metrics': os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    'metrics_flush_interval': int(os.getenv('METRICS_FLUSH_INTERVAL', '300'))
}