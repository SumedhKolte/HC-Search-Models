from pathlib import Path
from datetime import datetime, timezone
import requests
import logging

logger = logging.getLogger(__name__)

def get_ssl_cert_path() -> str:
    """Get SSL certificate path with auto-download and timestamp tracking"""
    project_root = Path(__file__).resolve().parent.parent.parent
    ssl_dir = project_root / 'ssl'
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
        logger.info("⬇️ Downloading RDS SSL certificate...")
        response = requests.get('https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem')
        response.raise_for_status()
        
        with open(cert_path, 'wb') as f:
            f.write(response.content)
        
        with open(timestamp_path, 'w') as f:
            f.write(datetime.now(timezone.utc).isoformat())
    
    return str(cert_path)