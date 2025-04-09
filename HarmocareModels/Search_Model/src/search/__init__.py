"""
Medical Search Package
Provides vector, text, and location-based search capabilities
"""

from .vector import VectorSearch
from .text import TextSearch
from .location import LocationSearch
from .ranking import ResultRanker

__all__ = [
    'VectorSearch',
    'TextSearch',
    'LocationSearch',
    'ResultRanker'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Medical Search Team'

# Import dependencies check
try:
    import faiss
    import numpy
    import torch
    from sentence_transformers import SentenceTransformer
    from geopy import distance, geocoders
except ImportError as e:
    raise ImportError(
        f"Missing required dependency: {str(e)}. "
        "Please install all requirements from requirements/base.txt"
    )

# Verify SSL certificate
from pathlib import Path
from datetime import datetime, timezone
import requests
import logging

logger = logging.getLogger(__name__)

def verify_ssl_cert():
    """Verify SSL certificate for AWS RDS"""
    try:
        ssl_dir = Path(__file__).parents[3] / 'ssl'
        ssl_dir.mkdir(exist_ok=True)
        cert_path = ssl_dir / 'rds-ca-bundle.pem'
        timestamp_path = ssl_dir / 'cert_timestamp.txt'
        
        needs_update = True
        if cert_path.exists() and timestamp_path.exists():
            with open(timestamp_path, 'r') as f:
                last_update = datetime.fromisoformat(f.read().strip())
                if (datetime.now(timezone.utc) - last_update).days < 30:
                    needs_update = False
        
        if needs_update:
            logger.info(f"Downloading SSL certificate at {datetime.now(timezone.utc).isoformat()}")
            response = requests.get('https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem')
            response.raise_for_status()
            
            with open(cert_path, 'wb') as f:
                f.write(response.content)
            
            with open(timestamp_path, 'w') as f:
                f.write(datetime.now(timezone.utc).isoformat())
        
        return str(cert_path)
        
    except Exception as e:
        logger.error(f"SSL certificate verification failed: {str(e)}")
        raise

# Initialize SSL certificate
SSL_CERT_PATH = verify_ssl_cert()