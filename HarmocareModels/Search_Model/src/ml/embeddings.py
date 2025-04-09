"""Embeddings generation module"""
from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from src.data.database import Database
from src.utils.logger import setup_logger
from config.settings import MODEL_CONFIG

logger = setup_logger(__name__)

def generate_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for given texts"""
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_tensor=False)
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

def update_entity_embeddings(db: Database, entity_type: str, batch_size: int = 32) -> Tuple[int, int]:
    """Update embeddings for an entity type"""
    try:
        # Get records without embeddings
        query = f"""
            SELECT * FROM {entity_type}
            WHERE embedding IS NULL
            LIMIT {batch_size}
        """
        records = db.execute_query(query)
        
        if not records:
            return 0, 0
            
        # Generate text representations
        texts = [
            f"{r.get('name', '')} {r.get('specialization', '')} {r.get('description', '')}"
            for r in records
        ]
        
        # Generate embeddings
        embeddings = generate_embeddings(texts)
        
        # Update database
        for idx, record in enumerate(records):
            db.execute(
                f"""
                UPDATE {entity_type}
                SET embedding = %s
                WHERE id = %s
                """,
                (embeddings[idx].tobytes(), record['id'])
            )
        
        return len(records), len(embeddings)
        
    except Exception as e:
        logger.error(f"Failed to update embeddings for {entity_type}: {str(e)}")
        raise

class EmbeddingGenerator:
    """Generate and manage embeddings for medical entities"""
    
    def __init__(self, model_name: str = MODEL_CONFIG['base_model']):
        """Initialize embedding generator"""
        self.model = SentenceTransformer(model_name)
        self.db = Database()
        self.batch_size = 32
        
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        try:
            return self.model.encode(texts, 
                                   batch_size=self.batch_size,
                                   show_progress_bar=True,
                                   convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise

    def get_entity_text(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Combine entity fields into searchable text"""
        if entity_type == 'doctors':
            return f"{entity['name']} {entity['specialization']} {entity['city']} {entity['degree']}"
        
        elif entity_type == 'hospitals':
            return f"{entity['name']} {entity['hospital_type']} {entity['location']}"
        
        elif entity_type == 'clinics':
            return f"{entity['name']} {entity['location']}"
        
        elif entity_type == 'diseases':
            common_names = ' '.join(entity['common_names']) if entity['common_names'] else ''
            tags = ' '.join(entity['tags']) if entity['tags'] else ''
            return f"{entity['name']} {common_names} {entity['description']} {tags}"
        
        elif entity_type == 'symptoms':
            tags = ' '.join(entity['tags']) if entity['tags'] else ''
            return f"{entity['name']} {entity['description']} {tags}"
        
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")

    def update_entity_embeddings(self, entity_type: str, batch_size: int = 100) -> None:
        """Update embeddings for entities that need updates"""
        try:
            # Get entities needing updates
            query = f"""
            SELECT *
            FROM {entity_type}
            WHERE embedding IS NULL
            OR updated_at > last_embedded_at
            """
            entities = self.db.execute_query(query)
            
            if not entities:
                logger.info(f"No {entity_type} need embedding updates")
                return
            
            logger.info(f"Updating embeddings for {len(entities)} {entity_type}")
            
            # Process in batches
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                
                # Generate text representations
                texts = [self.get_entity_text(entity, entity_type) for entity in batch]
                
                # Generate embeddings
                embeddings = self.generate_batch_embeddings(texts)
                
                # Update database
                for entity, embedding in zip(batch, embeddings):
                    self.db.execute(
                        f"""
                        UPDATE {entity_type}
                        SET embedding = %s,
                            last_embedded_at = %s
                        WHERE id = %s
                        """,
                        [embedding.tobytes(), datetime.now(timezone.utc), entity['id']]
                    )
            
            # Log success
            self.db.execute(
                """
                INSERT INTO model_training_metrics
                (entity_type, embeddings_count, dimension, executiontime, success, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    entity_type,
                    len(entities),
                    self.model.get_sentence_embedding_dimension(),
                    None,  # execution time
                    True,
                    None
                ]
            )
            
            logger.info(f"Successfully updated embeddings for {len(entities)} {entity_type}")
            
        except Exception as e:
            error_msg = f"Failed to update {entity_type} embeddings: {str(e)}"
            logger.error(error_msg)
            
            # Log failure
            self.db.execute(
                """
                INSERT INTO model_training_metrics
                (entity_type, embeddings_count, dimension, executiontime, success, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    entity_type,
                    0,
                    self.model.get_sentence_embedding_dimension(),
                    None,
                    False,
                    error_msg
                ]
            )
            raise

    def update_all_embeddings(self) -> None:
        """Update embeddings for all entity types"""
        entity_types = ['doctors', 'hospitals', 'clinics', 'diseases', 'symptoms']
        
        for entity_type in entity_types:
            try:
                self.update_entity_embeddings(entity_type)
            except Exception as e:
                logger.error(f"Failed to update {entity_type} embeddings: {str(e)}")
                continue