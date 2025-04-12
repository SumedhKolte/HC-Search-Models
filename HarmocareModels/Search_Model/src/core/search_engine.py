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
    VECTOR_DIM = 384  # Define the vector dimension as a constant at class level

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

        # Add keyword lists for entity detection
        self.hospital_keywords = {
            'hospital', 'multi-specialty', 'emergency', 'icu', 'care center',
            'medical center', '24x7', 'trauma', 'nursing home', 'healthcare',
            'medical facility', 'clinic', 'medicare', 'treatment center',
            'dialysis', 'isolation', 'mri', 'emergency room'
        }
        
        self.doctor_keywords = {
            'doctor', 'specialist', 'physician', 'surgeon', 'consultation',
            'dr', 'dr.', 'pediatrician', 'cardiologist', 'neurologist',
            'orthopedic', 'gynecologist', 'dermatologist', 'psychiatrist',
            'checkup', 'vaccination'
        }

    def infer_entity_type(self, query: str) -> str:
        """Detect whether query is looking for doctors or hospitals"""
        query_lower = query.lower()
        
        # Count keyword matches for each type
        hospital_matches = sum(1 for kw in self.hospital_keywords if kw in query_lower)
        doctor_matches = sum(1 for kw in self.doctor_keywords if kw in query_lower)
        
        # Check for specific hospital indicators
        hospital_indicators = [
            'icu bed', 'emergency room', 'casualty', '24x7', 'multi-specialty',
            'dialysis center', 'trauma center', 'medical center'
        ]
        if any(ind in query_lower for ind in hospital_indicators):
            return 'hospitals'
            
        # If more hospital keywords found
        if hospital_matches > doctor_matches:
            return 'hospitals'
        
        # Default to doctors if unclear
        return 'doctors'

    def _build_search_query(self, entity_type: str, embedding_str: str, id_column: str) -> str:
        """Build entity-specific search query"""
        if entity_type == 'hospitals':
            return f"""
                SELECT 
                    t.{id_column},
                    t.name,
                    t.hospital_type,
                    t.location,
                    t.rating,
                    t.establishment_year,
                    t.reg_no,
                    GREATEST(
                        1 - (t.embedding <-> '[{embedding_str}]'::vector({self.VECTOR_DIM})),
                        ts_rank_cd(t.search_vector, plainto_tsquery(:query))
                    ) AS similarity_score
                FROM {entity_type} t
                WHERE TRUE
            """
        elif entity_type == 'clinics':
            return f"""
                SELECT 
                    t.{id_column},
                    t.name,
                    t.location,
                    GREATEST(
                        1 - (t.embedding <-> '[{embedding_str}]'::vector({self.VECTOR_DIM})),
                        ts_rank_cd(t.search_vector, plainto_tsquery(:query))
                    ) AS similarity_score
                FROM {entity_type} t
                WHERE TRUE
            """
        else:  # doctors
            return f"""
                SELECT 
                    t.{id_column},
                    t.name,
                    t.specialization,
                    t.city,
                    t.rating,
                    t.experience,
                    GREATEST(
                        1 - (t.embedding <-> '[{embedding_str}]'::vector({self.VECTOR_DIM})),
                        ts_rank_cd(t.search_vector, plainto_tsquery(:query))
                    ) AS similarity_score
                FROM {entity_type} t
                WHERE TRUE
            """

    async def search_entity(self, entity_type: str, query: str, filters: Optional[Dict] = None, limit: int = 10) -> List[Dict]:
        """Search within a specific entity type"""
        try:
            normalized_query = query.strip().lower()
            query_embedding = self.model.encode(normalized_query)
            
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            id_column = self.id_mapping.get(entity_type)
            if not id_column:
                raise ValueError(f"Unknown entity type: {entity_type}")

            async with self.db.get_session() as session:
                embedding_str = ','.join(map(str, query_embedding))
                
                # Get base query for entity type
                base_query = self._build_search_query(entity_type, embedding_str, id_column)

                # Initialize filter parameters
                filter_conditions = []
                filter_params = {
                    "query": normalized_query,
                    "limit": limit
                }

                # Add location/city filter
                if filters and filters.get('city'):
                    location_column = 'location' if entity_type in ['hospitals', 'clinics'] else 'city'
                    filter_conditions.append(f"""
                        LOWER(TRIM(t.{location_column})) = LOWER(TRIM(:city))
                    """)
                    filter_params['city'] = filters['city'].strip()

                # Add filters to base query
                if filter_conditions:
                    base_query += " AND " + " AND ".join(filter_conditions)

                # Add similarity conditions
                base_query += f"""
                    AND (
                        (t.embedding IS NOT NULL 
                        AND (1 - (t.embedding <-> '[{embedding_str}]'::vector({self.VECTOR_DIM}))) > 0.001)
                        OR t.search_vector @@ plainto_tsquery(:query)
                    )
                    ORDER BY similarity_score DESC 
                    LIMIT :limit
                """

                # Execute query
                search_query = text(base_query)
                result = await session.execute(search_query, filter_params)
                
                return [dict(row) for row in result.mappings()]

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
            query = text(f"""
                SELECT
                    t.did,
                    t.name,
                    t.specialization,
                    t.city,
                    t.rating,
                    t.experience,
                    1 - (t.embedding <-> :embedding::vector({self.VECTOR_DIM})) AS similarity_score
                FROM doctors t
                WHERE t.embedding IS NOT NULL
                AND (1 - (t.embedding <-> :embedding::vector({self.VECTOR_DIM}))) > 0.001
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
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute vector search for hospitals"""
        try:
            query = text(f"""
                WITH hospital_scores AS (
                    SELECT
                        h.hid,
                        h.name,
                        h.hospital_type,
                        h.location,
                        h.rating,
                        h.facilities,
                        h.specialties,
                        ha.icu_beds_available,
                        ha.emergency_ready,
                        ha.oxygen_level,
                        1 - (h.embedding <-> :embedding::vector({self.VECTOR_DIM})) AS similarity_score,
                        earth_distance(
                            ll_to_earth(h.latitude, h.longitude),
                            ll_to_earth(:lat, :lon)
                        ) as distance_km
                    FROM hospitals h
                    LEFT JOIN hospital_availability ha ON h.hid = ha.hospital_id
                    WHERE h.embedding IS NOT NULL
                    AND (1 - (h.embedding <-> :embedding::vector({self.VECTOR_DIM}))) > 0.001
                )
                SELECT *,
                CASE 
                    WHEN distance_km < 5000 THEN similarity_score * 1.2
                    WHEN distance_km < 10000 THEN similarity_score * 1.1
                    ELSE similarity_score
                END as final_score
                FROM hospital_scores
                ORDER BY final_score DESC
                LIMIT :limit
            """)

            result = conn.execute(
                query,
                {
                    "embedding": embedding,
                    "lat": filters.get('lat', 0) if filters else 0,
                    "lon": filters.get('lon', 0) if filters else 0,
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
            query = text(f"""
                SELECT
                    t.disease_id,
                    t.name,
                    t.description,
                    t.severity,
                    t.treatments,
                    1 - (t.embedding <-> :embedding::vector({self.VECTOR_DIM})) AS similarity_score
                FROM diseases t
                WHERE t.embedding IS NOT NULL
                AND (1 - (t.embedding <-> :embedding::vector({self.VECTOR_DIM}))) > 0.001
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
                        cialization,
                    t.city,
                    t.rating,
                    t.experience,
                    1 - (t.embedding <-> '[{embedding_str}]'::vector({self.VECTOR_DIM})) AS similarity_score
                FROM {entity_type} t
                WHERE t.embedding IS NOT NULL
                AND (1 - (t.embedding <-> '[{embedding_str}]'::vector({self.VECTOR_DIM}))) > 0.001
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
                'result_types': kwargs.get('result_types'),
                'entity_types': kwargs.get('entity_types', []),
                'processed_query': kwargs.get('processed_query', ''),
                'filters': kwargs.get('filters', {}),
                'created_at': datetime.now(timezone.utc)
            }

            await self.metrics.log_search(**params)

        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            logger.error("Full traceback:", exc_info=True)

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



