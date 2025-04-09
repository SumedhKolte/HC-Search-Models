import logging
import logging.handlers
from pathlib import Path
from datetime import datetime, timezone
import json
from pythonjsonlogger import jsonlogger
import os
from typing import Dict, Any
import requests
from dotenv import load_dotenv
from config.logging_config import setup_logging
import logging.config

# Load environment variables
load_dotenv()

# Create monitoring directories
MONITORING_DIR = Path(__file__).parent.parent / 'monitoring'
LOGS_DIR = MONITORING_DIR / 'logs'
METRICS_DIR = MONITORING_DIR / 'metrics'
SSL_DIR = Path(__file__).parent.parent / 'ssl'

for dir_path in [LOGS_DIR, METRICS_DIR, SSL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Custom JSON formatter with timestamps
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add ISO format timestamps
        if not log_record.get('timestamp'):
            now = datetime.now(timezone.utc).isoformat()
            log_record['timestamp'] = now
        
        if not log_record.get('level'):
            log_record['level'] = record.levelname

# Download SSL certificate if needed
def ensure_ssl_cert() -> str:
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
                
            logging.info(f"SSL certificate updated at {timestamp.isoformat()}")
        except Exception as e:
            logging.error(f"Failed to update SSL certificate: {str(e)}")
            raise
    
    return str(cert_path)

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': LOGS_DIR / f'medical_search_{datetime.now(timezone.utc):%Y%m%d}.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Initialize logging with SSL certificate
def setup_logging():
    """Setup logging configuration with SSL certificate"""
    try:
        # Ensure SSL certificate exists
        ssl_cert_path = ensure_ssl_cert()
        
        # Update SSL path in environment
        os.environ['SSL_CERT_PATH'] = ssl_cert_path
        
        # Configure logging
        logging.config.dictConfig(LOGGING_CONFIG)
        
        # Log startup information
        logging.info(f"Logging initialized at {datetime.now(timezone.utc).isoformat()}")
        logging.info(f"SSL certificate path: {ssl_cert_path}")
        
    except Exception as e:
        print(f"Failed to setup logging: {str(e)}")
        raise

# Initialize logging when module is imported
setup_logging()

# Logger will be configured automatically on import
logger = logging.getLogger(__name__)