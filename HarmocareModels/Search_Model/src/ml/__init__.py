"""
Machine Learning package for medical search system
Provides embeddings generation, model training, and optimization
"""

import os
from pathlib import Path
from datetime import datetime, timezone
import requests
import logging
from typing import Dict, Optional
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MLConfig:
    """ML configuration and model management"""
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self):
        self.ssl_manager = SSLCertificateManager()
        self.cert_path = self.ssl_manager.setup_ssl()
        self.model_dir = Path(__file__).parents[2] / 'ml' / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.config = {
            'base_model': os.getenv('MODEL_NAME', self.DEFAULT_MODEL),
            'embedding_dim': int(os.getenv('EMBEDDING_DIM', self.EMBEDDING_DIM)),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': int(os.getenv('BATCH_SIZE', 32)),
            'max_seq_length': int(os.getenv('MAX_SEQ_LENGTH', 128))
        }

class SSLCertificateManager:
    """Manage SSL certificates for secure model downloads"""
    
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

class ModelManager:
    """Model loading and management"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = MLConfig() if config is None else config
        self._model = None
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of the model"""
        if self._model is None:
            self._model = SentenceTransformer(
                self.config.config['base_model'],
                device=self.config.config['device']
            )
            self._model.max_seq_length = self.config.config['max_seq_length']
        return self._model
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.config.config['embedding_dim']
    
    def get_device(self) -> str:
        """Get current device"""
        return self.config.config['device']

# Initialize configuration
ml_config = MLConfig()
model_manager = ModelManager(ml_config)

# Package exports
__all__ = ['MLConfig', 'ModelManager', 'model_manager', 'SSLCertificateManager']

# Package metadata
__version__ = '1.0.0'
__author__ = 'Medical Search Team'