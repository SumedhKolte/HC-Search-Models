from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import logging
import json
from tqdm import tqdm

from src.data.database import Database
from src.utils.logger import setup_logger
from config.settings import MODEL_CONFIG

logger = setup_logger(__name__)

class ModelTrainer:
    """Medical search model trainer"""
    
    def __init__(self, model_name: str = MODEL_CONFIG['base_model']):
        """Initialize trainer with model and configurations"""
        self.db = Database()
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.batch_size = MODEL_CONFIG['training']['batch_size']
        
    def prepare_training_data(self) -> List[InputExample]:
        """Prepare training data from search logs and entities"""
        try:
            # Get successful searches
            query = """
            SELECT 
                sl.query,
                sl.processed_query,
                sl.search_type,
                sl.results_count
            FROM search_logs sl
            WHERE sl.results_count > 0
            AND sl.created_at >= NOW() - INTERVAL '30 days'
            ORDER BY sl.log_id DESC
            LIMIT 10000
            """
            search_data = self.db.execute_query(query)
            
            # Prepare training examples
            examples = []
            for row in search_data:
                # Create positive pair
                examples.append(InputExample(
                    texts=[row['query'], row['processed_query']],
                    label=1.0
                ))
                
                # Get related entity text
                if row['search_type'] in ['doctor', 'hospital', 'clinic']:
                    entity_text = self._get_entity_text(row['search_type'], row['query'])
                    if entity_text:
                        examples.append(InputExample(
                            texts=[row['query'], entity_text],
                            label=1.0
                        ))
            
            logger.info(f"Prepared {len(examples)} training examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {str(e)}")
            raise

    def _get_entity_text(self, entity_type: str, query: str) -> Optional[str]:
        """Get relevant entity text for training"""
        try:
            # Add 's' to entity_type if not present
            if not entity_type.endswith('s'):
                entity_type += 's'
                
            query = f"""
            SELECT *
            FROM {entity_type}
            WHERE to_tsvector('english', 
                CASE 
                    WHEN name IS NOT NULL THEN name 
                    ELSE ''
                END ||
                CASE 
                    WHEN specialization IS NOT NULL THEN ' ' || specialization
                    ELSE ''
                END ||
                CASE 
                    WHEN city IS NOT NULL THEN ' ' || city
                    ELSE ''
                END
            ) @@ to_tsquery('english', %s)
            LIMIT 1
            """
            
            result = self.db.execute_query(query, [query])
            
            if result:
                entity = result[0]
                if entity_type == 'doctors':
                    return f"{entity['name']} {entity['specialization']} {entity['city']}"
                elif entity_type == 'hospitals':
                    return f"{entity['name']} {entity['hospital_type']} {entity['location']}"
                elif entity_type == 'clinics':
                    return f"{entity['name']} {entity['location']}"
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get entity text: {str(e)}")
            return None

    def train(self, 
             train_examples: List[InputExample],
             epochs: int = MODEL_CONFIG['training']['epochs'],
             evaluation_steps: int = MODEL_CONFIG['training']['evaluation_steps']) -> None:
        """Train the model on prepared examples"""
        try:
            start_time = datetime.now(timezone.utc)
            
            # Prepare data loader
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=self.batch_size
            )
            
            # Training loss
            train_loss = losses.CosineSimilarityLoss(self.model)
            
            # Train the model
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                evaluation_steps=evaluation_steps,
                warmup_steps=MODEL_CONFIG['training']['warmup_steps'],
                show_progress_bar=True
            )
            
            # Calculate execution time
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Log training metrics
            self.db.execute("""
                INSERT INTO model_training_metrics
                (entity_type, embeddings_count, dimension, executiontime, success, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, [
                'all',
                len(train_examples),
                self.model.get_sentence_embedding_dimension(),
                execution_time,
                True,
                None
            ])
            
            # Save the model
            self._save_model()
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg)
            
            # Log failure
            self.db.execute("""
                INSERT INTO model_training_metrics
                (entity_type, embeddings_count, dimension, executiontime, success, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, [
                'all',
                0,
                self.model.get_sentence_embedding_dimension(),
                None,
                False,
                error_msg
            ])
            raise

    def _save_model(self) -> None:
        """Save trained model with timestamp"""
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            save_dir = Path(MODEL_CONFIG['model_path']) / f'model_{timestamp}'
            
            # Save model
            self.model.save(str(save_dir))
            
            # Save training metadata
            metadata = {
                'timestamp': timestamp,
                'model_name': MODEL_CONFIG['base_model'],
                'embedding_dim': self.model.get_sentence_embedding_dimension()
            }
            
            with open(save_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Model saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise