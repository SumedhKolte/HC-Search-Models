#This file prevents 
# Search performance not to degrade over time
# Vector indexes not to become outdated
# Model parameters wouldn't be tuned to latest data
# System would miss opportunities for improved accuracy and speed

import sys
from pathlib import Path
import os
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
import faiss
import torch
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.database import Database
from src.ml.embeddings import generate_embeddings
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger
from config.settings import DB_CONFIG, MODEL_CONFIG

# Setup logging
logger = setup_logger(__name__)

def setup_ssl():
    """Setup SSL certificate for database connection"""
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

def optimize_embeddings(model, embeddings: np.ndarray, n_clusters: int = 100) -> faiss.Index:
    """Optimize embeddings using FAISS indexing"""
    dimension = embeddings.shape[1]
    
    # Normalize vectors
    faiss.normalize_L2(embeddings)
    
    # Create clustering index
    clustering_index = faiss.IndexFlatL2(dimension)
    
    # Create and train index
    index = faiss.IndexIVFFlat(clustering_index, dimension, n_clusters)
    index.train(embeddings)
    index.add(embeddings)
    
    return index

def optimize_model_parameters(model: SentenceTransformer, train_data: pd.DataFrame) -> Dict:
    """Optimize model hyperparameters"""
    results = {}
    learning_rates = [2e-5, 3e-5, 5e-5]
    batch_sizes = [16, 32, 64]
    
    best_metric = 0
    best_params = {}
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            params = {
                'learning_rate': lr,
                'batch_size': batch_size
            }
            
            metric = train_and_evaluate(model, train_data, params)
            results[f"lr_{lr}_batch_{batch_size}"] = metric
            
            if metric > best_metric:
                best_metric = metric
                best_params = params
    
    return best_params

def train_and_evaluate(model: SentenceTransformer, data: pd.DataFrame, params: Dict) -> float:
    """Train model with given parameters and evaluate performance"""
    try:
        # Training logic here
        metric = calculate_metrics(model, data)
        return metric
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        return 0.0

def update_indexes(db: Database, model: SentenceTransformer):
    """Update FAISS indexes for all entity types"""
    entity_types = ['doctors', 'hospitals', 'clinics', 'diseases', 'symptoms']
    
    for entity_type in entity_types:
        try:
            # Get embeddings
            query = f"SELECT * FROM {entity_type}"
            df = pd.read_sql(query, db.engine)
            
            # Generate new embeddings
            embeddings = generate_embeddings(model, df)
            
            # Optimize and save index
            index = optimize_embeddings(model, embeddings)
            index_path = project_root / 'ml' / 'indexes' / f"{entity_type}.index"
            faiss.write_index(index, str(index_path))
            
            logger.info(f"Updated index for {entity_type}")
            
        except Exception as e:
            logger.error(f"Error updating index for {entity_type}: {str(e)}")

def main():
    """Main optimization function"""
    logger.info("Starting optimization process")
    
    # Setup SSL
    ssl_cert_path = setup_ssl()
    
    try:
        # Initialize database connection
        db = Database(ssl_cert_path=ssl_cert_path)
        
        # Load model
        model = SentenceTransformer(MODEL_CONFIG['base_model'])
        
        # Get training data
        train_data = pd.read_sql("SELECT * FROM model_training_metrics", db.engine)
        
        # Optimize model parameters
        best_params = optimize_model_parameters(model, train_data)
        logger.info(f"Best parameters found: {best_params}")
        
        # Update indexes with optimized model
        update_indexes(db, model)
        
        # Log optimization metrics
        log_query = """
        INSERT INTO system_updates (status, error_message)
        VALUES (true, 'Optimization completed successfully')
        """
        db.engine.execute(log_query)
        
        logger.info("Optimization completed successfully")
        
    except Exception as e:
        error_msg = f"Optimization failed: {str(e)}"
        logger.error(error_msg)
        
        if 'db' in locals():
            db.engine.execute(
                "INSERT INTO system_updates (status, error_message) VALUES (false, %s)",
                [error_msg]
            )

if __name__ == "__main__":
    main()