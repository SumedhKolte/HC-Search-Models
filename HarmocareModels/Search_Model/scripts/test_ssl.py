# Without this script:

# Database connections might fail
# Security would be compromised
# System components could fail silently
# Compliance requirements might be violated

from pathlib import Path
import os
from dotenv import load_dotenv
import requests
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_ssl_setup():
    """Verify SSL certificate setup and database configuration"""
    
    # Load environment variables
    load_dotenv()
    
    # Setup SSL paths
    ssl_dir = Path(__file__).parent.parent / 'ssl'
    ssl_dir.mkdir(exist_ok=True)
    cert_path = ssl_dir / 'rds-ca-bundle.pem'
    
    # Download certificate if needed
    if not cert_path.exists():
        try:
            cert_url = 'https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem'
            response = requests.get(cert_url)
            response.raise_for_status()
            
            with open(cert_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"SSL certificate downloaded at: {datetime.now(timezone.utc).isoformat()}")
        except Exception as e:
            logger.error(f"Failed to download SSL certificate: {str(e)}")
            return False

    # Create symlink in PostgreSQL default location
    pg_ssl_dir = Path.home() / 'AppData' / 'Roaming' / 'postgresql'
    pg_ssl_dir.mkdir(parents=True, exist_ok=True)
    root_cert = pg_ssl_dir / 'root.crt'
    
    # Copy certificate to PostgreSQL location
    try:
        import shutil
        shutil.copy2(cert_path, root_cert)
        logger.info(f"Copied SSL certificate to PostgreSQL location: {root_cert}")
    except Exception as e:
        logger.error(f"Failed to copy SSL certificate: {str(e)}")
        return False
    
    # Verify database configuration
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'sslmode': 'verify-full',
        'sslrootcert': str(root_cert)  # Use the PostgreSQL default location
    }
    
    # Log configuration
    logger.info("Database Configuration:")
    for key, value in db_config.items():
        if key != 'password':  # Don't log the password
            logger.info(f"{key}: {value}")
    
    # Test database connection
    try:
        import psycopg2
        conn = psycopg2.connect(**db_config)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            logger.info(f"Successfully connected to database. PostgreSQL version: {version}")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return False

if __name__ == "__main__":
    success = verify_ssl_setup()
    logger.info(f"SSL setup verification {'successful' if success else 'failed'}")