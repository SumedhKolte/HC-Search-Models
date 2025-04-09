from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timezone
import logging
from sentence_transformers import util
import json

from src.data.database import Database
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ResultRanker:
    """Rank and score search results"""
    
    def __init__(self):
        """Initialize result ranker"""
        self.db = Database()
        self.weights = {
            'vector_score': 0.4,
            'text_score': 0.3,
            'rating_score': 0.2,
            'popularity_score': 0.1
        }
        
    def rank_results(self,
                    results: List[Dict],
                    entity_type: str,
                    query_embedding: np.ndarray,
                    query_text: str,
                    filters: Optional[Dict] = None) -> List[Dict]:
        """Rank results using multiple scoring factors"""
        try:
            start_time = datetime.now(timezone.utc)
            
            if not results:
                return []
                
            # Calculate scores
            scored_results = []
            for result in results:
                # Get individual scores
                vector_score = self._calculate_vector_score(
                    query_embedding, 
                    result.get('embedding')
                )
                
                text_score = self._calculate_text_score(
                    query_text,
                    result,
                    entity_type
                )
                
                rating_score = self._calculate_rating_score(result)
                popularity_score = self._calculate_popularity_score(
                    result,
                    entity_type
                )
                
                # Calculate weighted total score
                total_score = (
                    self.weights['vector_score'] * vector_score +
                    self.weights['text_score'] * text_score +
                    self.weights['rating_score'] * rating_score +
                    self.weights['popularity_score'] * popularity_score
                )
                
                scored_results.append({
                    **result,
                    'scores': {
                        'vector_score': float(vector_score),
                        'text_score': float(text_score),
                        'rating_score': float(rating_score),
                        'popularity_score': float(popularity_score),
                        'total_score': float(total_score)
                    }
                })
            
            # Sort by total score
            ranked_results = sorted(
                scored_results,
                key=lambda x: x['scores']['total_score'],
                reverse=True
            )
            
            # Log ranking metrics
            self._log_ranking_metrics(
                query=query_text,
                entity_type=entity_type,
                results_count=len(ranked_results),
                execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
            )
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Ranking failed: {str(e)}")
            return results

    def _calculate_vector_score(self,
                              query_embedding: np.ndarray,
                              result_embedding: Optional[bytes]) -> float:
        """Calculate vector similarity score"""
        try:
            if result_embedding is None:
                return 0.0
                
            # Convert binary embedding to numpy array
            result_vector = np.frombuffer(result_embedding, dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = util.cos_sim(
                query_embedding.reshape(1, -1),
                result_vector.reshape(1, -1)
            )
            
            return float(similarity[0][0])
            
        except Exception as e:
            logger.error(f"Vector score calculation failed: {str(e)}")
            return 0.0

    def _calculate_text_score(self,
                            query: str,
                            result: Dict,
                            entity_type: str) -> float:
        """Calculate text matching score"""
        try:
            # Get relevant text fields based on entity type
            if entity_type == 'doctors':
                text = f"{result.get('name', '')} {result.get('specialization', '')} {result.get('city', '')}"
            elif entity_type == 'hospitals':
                text = f"{result.get('name', '')} {result.get('hospital_type', '')} {result.get('location', '')}"
            elif entity_type == 'clinics':
                text = f"{result.get('name', '')} {result.get('location', '')}"
            else:
                text = result.get('name', '')
            
            # Calculate text similarity using TF-IDF weights
            query = f"""
            SELECT ts_rank_cd(to_tsvector('english', %s), to_tsquery('english', %s)) as rank
            """
            rank_result = self.db.execute_query(query, [text, query])
            
            return float(rank_result[0]['rank']) if rank_result else 0.0
            
        except Exception as e:
            logger.error(f"Text score calculation failed: {str(e)}")
            return 0.0

    def _calculate_rating_score(self, result: Dict) -> float:
        """Calculate normalized rating score"""
        try:
            rating = result.get('rating')
            if not rating:
                return 0.0
                
            # Convert rating to float and normalize to 0-1
            rating_float = float(rating)
            return min(max(rating_float / 5.0, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Rating score calculation failed: {str(e)}")
            return 0.0

    def _calculate_popularity_score(self, result: Dict, entity_type: str) -> float:
        """Calculate popularity score based on search patterns"""
        try:
            # Get search frequency for this entity
            query = """
            SELECT COUNT(*) as search_count
            FROM search_logs
            WHERE search_type = %s
            AND filters->>'id' = %s
            """
            
            entity_id = result.get('did' if entity_type == 'doctors' else 'hid' if entity_type == 'hospitals' else 'cid')
            count_result = self.db.execute_query(query, [entity_type, entity_id])
            
            if not count_result:
                return 0.0
                
            # Normalize by maximum searches
            max_query = """
            SELECT MAX(search_count) as max_searches
            FROM (
                SELECT COUNT(*) as search_count
                FROM search_logs
                WHERE search_type = %s
                GROUP BY filters->>'id'
            ) counts
            """
            max_result = self.db.execute_query(max_query, [entity_type])
            
            if not max_result or not max_result[0]['max_searches']:
                return 0.0
                
            return count_result[0]['search_count'] / max_result[0]['max_searches']
            
        except Exception as e:
            logger.error(f"Popularity score calculation failed: {str(e)}")
            return 0.0

    def _log_ranking_metrics(self,
                           query: str,
                           entity_type: str,
                           results_count: int,
                           execution_time: float) -> None:
        """Log ranking metrics"""
        try:
            self.db.execute("""
                INSERT INTO search_metrics
                (query, entity_type, total_results, execution_time, result_types)
                VALUES (%s, %s, %s, %s, %s)
            """, [
                query,
                entity_type,
                results_count,
                execution_time,
                json.dumps({'ranking_type': 'multi_factor'})
            ])
            
        except Exception as e:
            logger.error(f"Failed to log ranking metrics: {str(e)}")