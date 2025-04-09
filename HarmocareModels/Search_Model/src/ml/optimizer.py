#Help to optimize the model for medical search and improve its performance

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import optuna
from tqdm import tqdm
import json

from src.data.database import Database
from src.utils.logger import setup_logger
from config.settings import MODEL_CONFIG

logger = setup_logger(__name__)

class ModelOptimizer:
    """Optimize model performance for medical search"""
    
    def __init__(self, model_name: str = MODEL_CONFIG['base_model']):
        """Initialize model optimizer"""
        self.db = Database()
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def get_evaluation_data(self) -> Tuple[List[str], List[str]]:
        """Get evaluation data from search logs"""
        query = """
        SELECT 
            sl.query,
            sl.processed_query,
            sl.results_count,
            sl.search_type
        FROM search_logs sl
        WHERE sl.results_count > 0
        AND sl.search_type IN ('doctor', 'hospital', 'disease')
        ORDER BY log_id DESC
        LIMIT 1000
        """
        results = self.db.execute_query(query)
        
        queries = [row['query'] for row in results]
        processed_queries = [row['processed_query'] for row in results]
        
        return queries, processed_queries

    def evaluate_model(self, 
                      queries: List[str], 
                      processed_queries: List[str],
                      batch_size: int = 32) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Generate embeddings
            query_embeddings = self.model.encode(
                queries,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            processed_embeddings = self.model.encode(
                processed_queries,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Calculate similarities
            similarities = np.zeros((len(queries), len(processed_queries)))
            for i in range(len(queries)):
                similarities[i] = np.dot(processed_embeddings, query_embeddings[i]) / \
                                (np.linalg.norm(processed_embeddings, axis=1) * np.linalg.norm(query_embeddings[i]))
            
            # Calculate metrics
            precision = np.mean(similarities > 0.7)  # Similarity threshold
            recall = np.mean(similarities.max(axis=1) > 0.7)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'mean_similarity': float(np.mean(similarities)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise

    def optimize_hyperparameters(self, 
                               n_trials: int = 20, 
                               timeout: int = 3600) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna"""
        try:
            def objective(trial):
                # Hyperparameters to optimize
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
                batch_size = trial.suggest_int('batch_size', 16, 128)
                warmup_steps = trial.suggest_int('warmup_steps', 50, 500)
                
                # Update model parameters
                self.model.train_objectives[0][1].optimizer.param_groups[0]['lr'] = learning_rate
                
                # Get evaluation data
                queries, processed_queries = self.get_evaluation_data()
                train_queries, val_queries, train_processed, val_processed = train_test_split(
                    queries, processed_queries, test_size=0.2
                )
                
                # Train and evaluate
                metrics = self.evaluate_model(val_queries, val_processed, batch_size=batch_size)
                
                return metrics['f1_score']
            
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Log best parameters
            best_params = study.best_params
            best_params['best_value'] = study.best_value
            best_params['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Save optimization results
            self.db.execute("""
                INSERT INTO model_training_metrics
                (entity_type, embeddings_count, dimension, executiontime, success, error_message, optimization_params)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, [
                'all',
                0,
                self.model.get_sentence_embedding_dimension(),
                timeout,
                True,
                None,
                best_params
            ])
            
            return best_params
            
        except Exception as e:
            error_msg = f"Hyperparameter optimization failed: {str(e)}"
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

    def save_optimized_model(self, params: Dict[str, Any]) -> Path:
        """Save optimized model with parameters"""
        try:
            # Update model parameters
            self.model.train_objectives[0][1].optimizer.param_groups[0]['lr'] = params['learning_rate']
            
            # Save model
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            save_dir = Path(MODEL_CONFIG['model_path']) / f'optimized_{timestamp}'
            self.model.save(str(save_dir))
            
            # Save parameters
            params_file = save_dir / 'optimization_params.json'
            with open(params_file, 'w') as f:
                json.dump(params, f)
            
            return save_dir
            
        except Exception as e:
            logger.error(f"Failed to save optimized model: {str(e)}")
            raise