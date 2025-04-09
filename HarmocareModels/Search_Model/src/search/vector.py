from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timezone
import logging
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.data.database import Database
from src.utils.logger import setup_logger
from config.settings import MODEL_CONFIG

logger = setup_logger(__name__)

class VectorSearch:
    """Vector-based similarity search using FAISS indexes"""
    
    def __init__(self):
        """Initialize vector search"""
        self.db = Database()
        self.model = SentenceTransformer(MODEL_CONFIG['base_model'])
        self.index_dir = Path(MODEL_CONFIG['index_path'])
        self.index_dir.mkdir(exist_ok=True)
        self.indexes = {}
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize indexes
        self._load_indexes()
        
    def search(self,
              query: str,
              entity_type: str,
              k: int = 10,
              threshold: float = 0.7,
              filters: Optional[Dict] = None) -> List[Dict]:
        """Perform vector similarity search"""
        try:
            start_time = datetime.now(timezone.utc)
            
            # Generate query embedding
            query_embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Get results from FAISS
            if entity_type not in self.indexes:
                logger.error(f"No index found for entity type: {entity_type}")
                return []
                
            distances, indices = self.indexes[entity_type].search(
                query_embedding.reshape(1, -1).astype('float32'),
                k
            )
            
            # Get entity details
            results = self._get_entity_details(
                entity_type=entity_type,
                indices=indices[0],
                distances=distances[0],
                threshold=threshold,
                filters=filters
            )
            
            # Log search metrics
            self._log_search_metrics(
                query=query,
                entity_type=entity_type,
                results_count=len(results),
                execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def _load_indexes(self) -> None:
        """Load FAISS indexes for each entity type"""
        try:
            entity_types = ['doctors', 'hospitals', 'clinics', 'diseases', 'symptoms']
            
            for entity_type in entity_types:
                index_path = self.index_dir / f"{entity_type}_index.faiss"
                
                if index_path.exists():
                    self.indexes[entity_type] = faiss.read_index(str(index_path))
                else:
                    # Create new index
                    index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                    self._build_index(entity_type, index)
                    self.indexes[entity_type] = index
                    
                    # Save index
                    faiss.write_index(index, str(index_path))
                    
        except Exception as e:
            logger.error(f"Failed to load indexes: {str(e)}")
            raise

    def _build_index(self, entity_type: str, index: faiss.Index) -> None:
        """Build FAISS index for entity type"""
        try:
            # Get embeddings from database
            query = f"""
            SELECT embedding
            FROM {entity_type}
            WHERE embedding IS NOT NULL
            """
            results = self.db.execute_query(query)
            
            if not results:
                logger.warning(f"No embeddings found for {entity_type}")
                return
                
            # Convert binary embeddings to numpy array
            embeddings = np.stack([
                np.frombuffer(result['embedding'], dtype=np.float32)
                for result in results
            ])
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            
            # Add to index
            index.add(embeddings)
            
            logger.info(f"Built index for {entity_type} with {len(results)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to build index for {entity_type}: {str(e)}")
            raise

    def _get_entity_details(self,
                          entity_type: str,
                          indices: np.ndarray,
                          distances: np.ndarray,
                          threshold: float,
                          filters: Optional[Dict]) -> List[Dict]:
        """Get entity details for search results"""
        try:
            # Filter by similarity threshold
            valid_idx = np.where(distances >= threshold)[0]
            if len(valid_idx) == 0:
                return []
                
            indices = indices[valid_idx]
            distances = distances[valid_idx]
            
            # Build query based on entity type
            if entity_type == 'doctors':
                query = """
                SELECT 
                    d.*,
                    h.name as hospital_name,
                    c.name as clinic_name
                FROM doctors d
                LEFT JOIN hospitals h ON d.hid = h.hid
                LEFT JOIN clinics c ON d.cid = c.cid
                WHERE d.did = ANY(%s)
                """
            else:
                query = f"""
                SELECT *
                FROM {entity_type}
                WHERE {entity_type[:-1]}_id = ANY(%s)
                """
            
            # Get entity details
            results = self.db.execute_query(query, [list(indices)])
            
            # Add similarity scores
            scored_results = []
            for result, distance in zip(results, distances):
                result['similarity_score'] = float(distance)
                scored_results.append(result)
            
            # Apply filters
            if filters:
                scored_results = [
                    result for result in scored_results
                    if all(result.get(k) == v for k, v in filters.items())
                ]
            
            return scored_results
            
        except Exception as e:
            logger.error(f"Failed to get entity details: {str(e)}")
            return []

    def _log_search_metrics(self,
                          query: str,
                          entity_type: str,
                          results_count: int,
                          execution_time: float) -> None:
        """Log vector search metrics"""
        try:
            self.db.execute("""
                INSERT INTO search_metrics
                (query, total_results, execution_time, result_types)
                VALUES (%s, %s, %s, %s)
            """, [
                query,
                results_count,
                execution_time,
                json.dumps({
                    'type': 'vector_search',
                    'entity_type': entity_type
                })
            ])
            
        except Exception as e:
            logger.error(f"Failed to log search metrics: {str(e)}")