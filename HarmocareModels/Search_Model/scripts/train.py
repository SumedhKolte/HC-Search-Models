#!/usr/bin/env python3
import sys
from pathlib import Path
import os
from datetime import datetime, timezone
import requests
import logging
import json
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.database import Database
from src.ml.embeddings import generate_embeddings
from src.utils.logger import setup_logger
from config.settings import DB_CONFIG, MODEL_CONFIG

# Setup logging
logger = setup_logger(__name__)

def setup_ssl() -> str:
    """Setup SSL certificate with timestamp tracking"""
    ssl_dir = project_root / 'ssl'
    ssl_dir.mkdir(exist_ok=True)
    cert_path = ssl_dir / 'rds-ca-bundle.pem'
    timestamp_path = ssl_dir / 'cert_timestamp.txt'
    
    current_time = datetime.now(timezone.utc)
    
    needs_update = True
    if cert_path.exists() and timestamp_path.exists():
        with open(timestamp_path, 'r') as f:
            last_update = datetime.fromisoformat(f.read().strip())
            if (current_time - last_update).days < 30:
                needs_update = False
    
    if needs_update:
        logger.info(f"Downloading SSL certificate at {current_time.isoformat()}")
        response = requests.get('https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem')
        response.raise_for_status()
        
        with open(cert_path, 'wb') as f:
            f.write(response.content)
        
        with open(timestamp_path, 'w') as f:
            f.write(current_time.isoformat())
    
    return str(cert_path)

def get_training_data(db: Database) -> Dict[str, List]:
    """Get training data from database"""
    data = {}
    
    # Get doctors data
    query = """
    SELECT name, specialization, city, degree, search_vector
    FROM doctors 
    WHERE search_vector IS NOT NULL
    """
    doctors = db.execute_query(query)
    data['doctors'] = doctors
    
    # Get hospitals data
    query = """
    SELECT name, hospital_type, location, search_vector
    FROM hospitals
    WHERE search_vector IS NOT NULL
    """
    hospitals = db.execute_query(query)
    data['hospitals'] = hospitals
    
    # Get diseases data
    query = """
    SELECT name, description, common_names, search_vector
    FROM diseases
    WHERE search_vector IS NOT NULL
    """
    diseases = db.execute_query(query)
    data['diseases'] = diseases
    
    return data

def train_model(
    data: Dict[str, List],
    model_name: str = MODEL_CONFIG['base_model'],
    output_dir: Optional[Path] = None
) -> SentenceTransformer:
    """Train the model on medical search data"""
    
    if output_dir is None:
        output_dir = project_root / 'ml' / 'models' / f"medical_search_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = SentenceTransformer(model_name)
    
    # Prepare training data
    train_examples = []
    for entity_type, entities in data.items():
        for entity in entities:
            text = ' '.join(str(v) for v in entity.values() if v is not None)
            train_examples.append(text)
    
    # Split data
    train_data, eval_data = train_test_split(
        train_examples,
        test_size=0.2,
        random_state=42
    )
    
    # Create data loader
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=MODEL_CONFIG['training']['batch_size']
    )
    
    # Train the model
    train_start = datetime.now(timezone.utc)
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, losses.MultipleNegativesRankingLoss(model))],
            epochs=MODEL_CONFIG['training']['epochs'],
            evaluation_steps=MODEL_CONFIG['training']['evaluation_steps'],
            output_path=str(output_dir),
            show_progress_bar=True
        )
        
        train_end = datetime.now(timezone.utc)
        execution_time = (train_end - train_start).total_seconds()
        
        # Log training metrics
        metrics = {
            'entity_type': 'all',
            'embeddings_count': len(train_examples),
            'dimension': model.get_sentence_embedding_dimension(),
            'execution_time': execution_time,
            'success': True,
            'error_message': None
        }
        
        return model, metrics
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        
        metrics = {
            'entity_type': 'all',
            'embeddings_count': len(train_examples),
            'dimension': model.get_sentence_embedding_dimension(),
            'execution_time': None,
            'success': False,
            'error_message': error_msg
        }
        
        return None, metrics

def main():
    """Main training function"""
    logger.info("Starting model training process")
    
    # Setup SSL
    ssl_cert_path = setup_ssl()
    
    try:
        # Initialize database connection
        db = Database(ssl_cert_path=ssl_cert_path)
        
        # Get training data
        training_data = get_training_data(db)
        
        # Train model
        model, metrics = train_model(training_data)
        
        if model:
            # Generate new embeddings
            for entity_type, entities in training_data.items():
                generate_embeddings(model, entities, entity_type)
            
            # Log success
            logger.info("Training completed successfully")
            db.execute(
                "INSERT INTO system_updates (status, error_message) VALUES (true, 'Training completed successfully')"
            )
        
        # Log metrics
        db.execute(
            """
            INSERT INTO model_training_metrics 
            (entity_type, embeddings_count, dimension, executiontime, success, error_message)
            VALUES (%(entity_type)s, %(embeddings_count)s, %(dimension)s, %(execution_time)s, 
                    %(success)s, %(error_message)s)
            """,
            metrics
        )
            
    except Exception as e:
        error_msg = f"Training process failed: {str(e)}"
        logger.error(error_msg)
        
        if 'db' in locals():
            db.execute(
                "INSERT INTO system_updates (status, error_message) VALUES (false, %s)",
                [error_msg]
            )

if __name__ == "__main__":
    main()