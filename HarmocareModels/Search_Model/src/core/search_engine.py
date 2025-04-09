from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timezone
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError
from sentence_transformers import SentenceTransformer
from src.core.search_builder import SearchQueryBuilder
from src.data.database import Database
from src.utils.metrics import PerformanceMetrics
from src.nlp.processor import QueryProcessor
import numpy as np

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self):  # Fixed method name from _init_ to __init__
        self.db = Database()
        self.query_builder = SearchQueryBuilder()
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.metrics = PerformanceMetrics()
        self.query_processor = QueryProcessor()
        self.id_mapping = {
            'doctors': 'did',
            'hospitals': 'hid',
            'clinics': 'cid',
            'diseases': 'disease_id',
            'symptoms': 'symptom_id'
        }

    async def search_entity(self, entity_type: str, query: str, limit: int = 10) -> List[Dict]:
        """Search within a specific entity type"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Get correct ID column
            id_column = self.id_mapping.get(entity_type)
            if not id_column:
                raise ValueError(f"Unknown entity type: {entity_type}")

            # Execute search using async session
            async with self.db.get_session() as session:
                # Format embedding as string
                embedding_str = ','.join(map(str, query_embedding))
                
                # Combined vector and text search query using named parameters
                search_query = text(f"""
                    SELECT 
                        t.{id_column},
                        t.name,
                        t.specialization, 
                        t.city,
                        t.rating,
                        t.experience,
                        GREATEST(
                            1 - (t.embedding <-> '[{embedding_str}]'::vector(384)),
                            ts_rank_cd(t.search_vector, plainto_tsquery(:query))
                        ) AS similarity_score
                    FROM {entity_type} t
                    WHERE (
                        t.embedding IS NOT NULL 
                        OR t.search_vector @@ plainto_tsquery(:query)
                    )
                    ORDER BY similarity_score DESC
                    LIMIT :limit
                """)

                # Pass parameters as a dictionary
                params = {
                    "query": query,
                    "limit": limit
                }

                result = await session.execute(search_query, params)
                results = [dict(row) for row in result.mappings()]
                return self._process_results(results, entity_type, id_column)

        except Exception as e:
            logger.error(f"Search failed for {entity_type}: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return []

    def _search_doctors_by_embedding(
        self,
        conn,
        embedding: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute vector search for doctors"""
        try:
            query = text("""
                SELECT
                    t.did,
                    t.name,
                    t.specialization,
                    t.city,
                    t.rating,
                    t.experience,
                    1 - (t.embedding <-> :embedding::vector(384)) AS similarity_score
                FROM doctors t
                WHERE t.embedding IS NOT NULL
                AND (1 - (t.embedding <-> :embedding::vector(384))) > 0.001
                ORDER BY similarity_score DESC
                LIMIT :limit
            """)

            result = conn.execute(
                query,
                {
                    "embedding": embedding,
                    "limit": limit
                }
            ).mappings().all()

            return result

        except SQLAlchemyError as e:
            logger.error(f"❌ Error in doctor vector search: {str(e)}")
            return []

    def _search_hospitals_by_embedding(
        self,
        conn,
        embedding: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute vector search for hospitals"""
        try:
            query = text("""
                SELECT
                    t.hid,
                    t.name,
                    t.hospital_type,
                    t.location,
                    t.rating,
                    1 - (t.embedding <-> :embedding::vector(384)) AS similarity_score
                FROM hospitals t
                WHERE t.embedding IS NOT NULL
                AND (1 - (t.embedding <-> :embedding::vector(384))) > 0.001
                ORDER BY similarity_score DESC
                LIMIT :limit
            """)

            result = conn.execute(
                query,
                {
                    "embedding": embedding,
                    "limit": limit
                }
            ).mappings().all()

            return result

        except SQLAlchemyError as e:
            logger.error(f"❌ Error in hospital vector search: {str(e)}")
            return []

    def _search_diseases_by_embedding(
        self,
        conn,
        embedding: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute vector search for diseases"""
        try:
            query = text("""
                SELECT
                    t.disease_id,
                    t.name,
                    t.description,
                    t.severity,
                    t.treatments,
                    1 - (t.embedding <-> :embedding::vector(384)) AS similarity_score
                FROM diseases t
                WHERE t.embedding IS NOT NULL
                AND (1 - (t.embedding <-> :embedding::vector(384))) > 0.001
                ORDER BY similarity_score DESC
                LIMIT :limit
            """)

            result = conn.execute(
                query,
                {
                    "embedding": embedding,
                    "limit": limit
                }
            ).mappings().all()

            return result

        except SQLAlchemyError as e:
            logger.error(f"❌ Error in disease vector search: {str(e)}")
            return []

    def _execute_search(
        self, 
        conn, 
        query_embedding: List[float],
        query_text: str,
        entity_type: str,
        id_column: str,
        limit: int
    ) -> List[Dict]:
        """Execute vector search with fallback to full-text"""
        try:
            logger.info(f"🔍 Performing vector search for: '{query_text}' in '{entity_type}'")

            # Vector similarity search with proper parameter binding
            embedding_str = ','.join(map(str, query_embedding))
            vector_query = text(f"""
                SELECT
                    t.{id_column},
                    t.name,
                    t.specialization,
                    t.city,
                    t.rating,
                    t.experience,
                    1 - (t.embedding <-> '[{embedding_str}]'::vector(384)) AS similarity_score
                FROM {entity_type} t
                WHERE t.embedding IS NOT NULL
                AND (1 - (t.embedding <-> '[{embedding_str}]'::vector(384))) > 0.001
                ORDER BY similarity_score DESC
                LIMIT :limit
            """)

            result = conn.execute(
                vector_query,
                {
                    "limit": limit
                }
            ).mappings().all()

            if result:
                logger.info(f"✅ Vector search found {len(result)} results")
                return result

            # Fallback to full-text search if no vector results
            logger.info("⚠ No vector results, falling back to full-text search...")

            fallback_query = text(f"""
                SELECT
                    t.{id_column},
                    t.name,
                    t.specialization,
                    t.city,
                    t.rating,
                    t.experience,
                    ts_rank_cd(t.search_vector, plainto_tsquery(:query)) AS similarity_score
                FROM {entity_type} t
                WHERE t.search_vector @@ plainto_tsquery(:query)
                ORDER BY similarity_score DESC
                LIMIT :limit
            """)

            fallback_result = conn.execute(
                fallback_query,
                {
                    "query": query_text,
                    "limit": limit
                }
            ).mappings().all()

            if fallback_result:
                logger.info(f"✅ Full-text search found {len(fallback_result)} results")
            else:
                logger.warning("❌ No results found in either search method")

            return fallback_result

        except SQLAlchemyError as e:
            logger.error(f"❌ Search execution failed: {str(e)}")
            logger.exception("Full traceback:")
            raise

    def _process_results(self, results: List[Dict], entity_type: str, id_column: str) -> List[Dict]:
        """Process and format search results"""
        try:
            return [
                {
                    'id': str(row[id_column]),
                    'type': entity_type,
                    'name': row['name'],
                    'score': float(row['similarity_score']),
                    'metadata': {
                        'specialization': row.get('specialization', ''),
                        'city': row.get('city', ''),
                        'rating': row.get('rating', '0'),
                        'experience': row.get('experience', '')
                    }
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return []

    def _prepare_tsquery(self, query: str) -> str:
        """Prepare query for PostgreSQL full-text search"""
        # Clean and normalize query
        cleaned = ' & '.join(
            word for word in query.replace("'", "").split()
            if len(word) > 2
        )
        return cleaned or query  # Fallback to original if cleaning removes everything

    async def search(
        self,
        query: str,
        entity_types: List[str],
        filters: Optional[Dict] = None
    ) -> Dict:
        """Execute search across specified entity types"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            all_results = []
            for entity_type in entity_types:
                try:
                    # Build and execute search query
                    search_query = self.query_builder.build_vector_query(
                        entity_type=entity_type,
                        query_embedding=query_embedding,
                        filters=filters
                    )
                    
                    # Execute search with proper row conversion
                    with self.db.engine.connect() as conn:
                        result = conn.execute(
                            search_query['query'],
                            search_query['params']
                        ).mappings().all()  # Use mappings() for proper dict conversion
                        
                        # Process results
                        for row in result:
                            id_key = f'{entity_type[:-1]}_id'
                            
                            result_item = {
                                'id': str(row.get(id_key, '')),
                                'type': entity_type,
                                'name': row.get('name', ''),
                                'score': float(row.get('similarity_score', 0)),
                                'metadata': {
                                    k: v for k, v in row.items()
                                    if k not in ['embedding', 'similarity_score', id_key, 'name']
                                }
                            }
                            all_results.append(result_item)

                except Exception as e:
                    logger.error(f"Search failed for {entity_type}: {str(e)}")
                    continue

            # Sort by similarity score
            all_results.sort(key=lambda x: x['score'], reverse=True)

            return {
                'status': 'success',
                'data': all_results,
                'metadata': {
                    'processed_query': query,
                    'entity_types': entity_types,
                    'total_results': len(all_results)
                }
            }

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'data': []
            }

    async def _log_search_metrics(self, **kwargs):
        """Log search metrics"""
        try:
            params = {
                'query': kwargs['query'],
                'total_results': kwargs['total_results'],
                'execution_time': kwargs['execution_time'],
                'result_types': ','.join(kwargs['entity_types']),
                'processed_query': kwargs.get('processed_query', ''),
                'filters': kwargs.get('filters', {}),
                'created_at': datetime.now(timezone.utc)
            }

            await self.metrics.log_search(**params)

        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")

    async def _search_entity(
        self,
        entity_type: str,
        query_embedding: List[float],
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search within a specific entity type using vector similarity
        """
        try:
            # Build search query using query builder
            search_query = self.query_builder.build_vector_query(
                entity_type=entity_type,
                embedding=query_embedding,
                filters=filters
            )

            # Execute search query
            results = self.db.fetch_all(
                query=search_query['query'],
                params=search_query['params']
            )

            # Process and format results
            processed_results = []
            for row in results:
                processed_results.append({
                    'id': row[f'{entity_type}_id'],
                    'type': entity_type,
                    'name': row['name'],
                    'description': row.get('description', ''),
                    'score': float(row['similarity_score']),
                    'metadata': {
                        k: v for k, v in row.items()
                        if k not in ['embedding', 'similarity_score', f'{entity_type}_id', 'name', 'description']
                    }
                })

            return processed_results

        except Exception as e:
            logger.error(f"Entity search failed for {entity_type}: {str(e)}")
            return []