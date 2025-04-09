"""
Utilities package for medical search system
Provides logging, metrics, validation and SSL certificate management
"""

from .logger import setup_logger
from .metrics import PerformanceMetrics
from .validators import InputValidator
from datetime import datetime, timezone
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# Package metadata
__version__ = '1.0.0'
__author__ = 'Medical Search Team'

def setup_ssl_certificate() -> str:
    """Setup and verify SSL certificate for AWS RDS"""
    try:
        ssl_dir = Path(__file__).parents[2] / 'ssl'
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
                
            logger.info(f"SSL certificate updated successfully at {datetime.now(timezone.utc).isoformat()}")
        
        return str(cert_path)
        
    except Exception as e:
        logger.error(f"SSL certificate setup failed: {str(e)}")
        raise

# Initialize SSL certificate on import
SSL_CERT_PATH = setup_ssl_certificate()

__all__ = [
    'PerformanceMetrics'
]