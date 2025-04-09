#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import datetime, timezone
import logging
import numpy as np
import faiss
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import json
import requests
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.database import Database
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

def get_entity_embeddings(db: Database, entity_type: str) -> tuple:
    """Get entity embeddings from database"""
    query = f"""
    SELECT id, embedding 
    FROM {entity_type}
    WHERE embedding IS NOT NULL
    """
    results = db.execute_query(query)
    
    ids = [r['id'] for r in results]
    embeddings = np.stack([np.frombuffer(r['embedding'], dtype=np.float32) for r in results])
    
    return ids, embeddings

def create_faiss_index(embeddings: np.ndarray, index_type: str = 'IVFFlat') -> faiss.Index:
    """Create FAISS index for vector search"""
    dimension = embeddings.shape[1]
    
    if index_type == 'IVFFlat':
        # Number of clusters - rule of thumb: sqrt(N)
        nlist = int(np.sqrt(len(embeddings)))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors per node
        
    else:
        index = faiss.IndexFlatL2(dimension)
    
    # Train index if needed
    if isinstance(index, faiss.IndexIVF):
        index.train(embeddings)
    
    # Add vectors to index
    index.add(embeddings)
    
    return index

def save_index(index: faiss.Index, entity_type: str, ids: List):
    """Save FAISS index and IDs mapping"""
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    index_dir = project_root / 'ml' / 'indexes' / entity_type
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = index_dir / f'index_{timestamp}.faiss'
    faiss.write_index(index, str(index_path))
    
    # Save ID mapping
    mapping_path = index_dir / f'id_mapping_{timestamp}.json'
    with open(mapping_path, 'w') as f:
        json.dump({'ids': ids}, f)
    
    # Update latest symlinks
    latest_index = index_dir / 'latest.faiss'
    latest_mapping = index_dir / 'latest_mapping.json'
    
    if latest_index.exists():
        latest_index.unlink()
    if latest_mapping.exists():
        latest_mapping.unlink()
    
    latest_index.symlink_to(index_path)
    latest_mapping.symlink_to(mapping_path)
    
    return index_path, mapping_path

def main():
    """Main index update function"""
    logger.info("Starting index update process")
    
    # Setup SSL
    ssl_cert_path = setup_ssl()
    
    try:
        # Initialize database connection
        db = Database(ssl_cert_path=ssl_cert_path)
        
        # Entity types to index
        entity_types = ['doctors', 'hospitals', 'clinics', 'diseases', 'symptoms']
        
        for entity_type in tqdm(entity_types, desc="Updating indexes"):
            try:
                # Get embeddings
                ids, embeddings = get_entity_embeddings(db, entity_type)
                
                if len(embeddings) == 0:
                    logger.warning(f"No embeddings found for {entity_type}")
                    continue
                
                # Create index
                index = create_faiss_index(embeddings)
                
                # Save index and mapping
                index_path, mapping_path = save_index(index, entity_type, ids)
                
                logger.info(f"Updated index for {entity_type}: {index_path}")
                
                # Log success
                db.execute(
                    """
                    INSERT INTO system_updates (status, error_message) 
                    VALUES (true, %s)
                    """,
                    [f"Index updated successfully for {entity_type}"]
                )
                
            except Exception as e:
                error_msg = f"Failed to update index for {entity_type}: {str(e)}"
                logger.error(error_msg)
                
                db.execute(
                    """
                    INSERT INTO system_updates (status, error_message)
                    VALUES (false, %s)
                    """,
                    [error_msg]
                )
                
    except Exception as e:
        error_msg = f"Index update process failed: {str(e)}"
        logger.error(error_msg)
        
        if 'db' in locals():
            db.execute(
                "INSERT INTO system_updates (status, error_message) VALUES (false, %s)",
                [error_msg]
            )

if __name__ == "__main__":
    main()