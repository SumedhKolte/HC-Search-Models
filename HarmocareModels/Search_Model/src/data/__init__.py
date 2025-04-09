"""
Data package for medical search system
Provides database connectivity and SSL certificate management
"""

from pathlib import Path
from datetime import datetime, timezone
import requests
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

# SSL Certificate Management
class SSLCertificateManager:
    """Manage SSL certificates for database connections"""
    
    def __init__(self):
        self.ssl_dir = Path(__file__).parents[2] / 'ssl'
        self.ssl_dir.mkdir(exist_ok=True)
        self.cert_path = self.ssl_dir / 'rds-ca-bundle.pem'
        self.timestamp_path = self.ssl_dir / 'cert_timestamp.txt'

    def setup_ssl(self) -> str:
        """Setup and verify SSL certificate"""
        try:
            needs_update = True
            if self.cert_path.exists() and self.timestamp_path.exists():
                with open(self.timestamp_path, 'r') as f:
                    last_update = datetime.fromisoformat(f.read().strip())
                    if (datetime.now(timezone.utc) - last_update).days < 30:
                        needs_update = False
            
            if needs_update:
                current_time = datetime.now(timezone.utc)
                logger.info(f"Downloading SSL certificate at {current_time.isoformat()}")
                
                response = requests.get('https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem')
                response.raise_for_status()
                
                with open(self.cert_path, 'wb') as f:
                    f.write(response.content)
                
                with open(self.timestamp_path, 'w') as f:
                    f.write(current_time.isoformat())
                
                logger.info("SSL certificate updated successfully")
            
            return str(self.cert_path)
            
        except Exception as e:
            logger.error(f"SSL certificate setup failed: {str(e)}")
            raise

# Database configuration
class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self):
        self.ssl_manager = SSLCertificateManager()
        self.cert_path = self.ssl_manager.setup_ssl()
        
        # Connection parameters
        self.params = {
            'host': os.getenv('DB_HOST'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'port': os.getenv('DB_PORT', '5432'),
            'sslmode': 'verify-full',
            'sslcert': self.cert_path
        }
        
    @property
    def connection_string(self) -> str:
        """Get database connection string"""
        return (
            f"postgresql://{self.params['user']}:{self.params['password']}"
            f"@{self.params['host']}:{self.params['port']}/{self.params['database']}"
        )

# Package exports
__all__ = ['SSLCertificateManager', 'DatabaseConfig']

# Initialize package-level configurations
ssl_manager = SSLCertificateManager()
db_config = DatabaseConfig()

# Package metadata
__version__ = '1.0.0'
__author__ = 'Medical Search Team'