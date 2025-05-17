import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
import json
import faiss
import os
from dotenv import load_dotenv

from src.data.database import Database
from src.nlp.processor import TextProcessor
from src.nlp.expander import QueryExpander
from src.utils.logger import setup_logger
from src.core.indexer import IndexManager
from config.settings import MODEL_CONFIG
from src.utils.validators import InputValidator
from src.utils.constants import EMBEDDING_MODELS

logger = setup_logger(__name__)

load_dotenv()

class SearchEngine:
    """Core search engine implementation"""
    
    def __init__(self):
        """Initialize search engine with model"""
        try:
            # Use all-MiniLM-L6-v2 as default model
            model_name = os.getenv('MODEL_NAME', EMBEDDING_MODELS['default'])
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = 384  # Fixed dimension for all-MiniLM-L6-v2
        except Exception as e:
            logger.error(f"Error initializing search engine: {str(e)}")
            raise

        self.db = Database()
        self.index_manager = IndexManager()
        self.text_processor = TextProcessor()
        self.query_expander = QueryExpander()
        self.validator = InputValidator()

    async def _load_medical_specialties(self) -> List[Dict]:
        """Load medical specialties from database"""
        try:
            query = """
            SELECT DISTINCT specialization 
            FROM doctors 
            WHERE specialization IS NOT NULL
            ORDER BY specialization
            """
            async with self.db.get_session() as session:
                result = await session.execute(text(query))
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to load specialties: {str(e)}")
            return []

    async def _load_medical_symptoms(self) -> List[Dict]:
        """Load medical symptoms from database"""
        try:
            query = """
            SELECT symptom_id, name, description
            FROM symptoms
            WHERE name IS NOT NULL
            ORDER BY name
            """
            async with self.db.get_session() as session:
                result = await session.execute(text(query))
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to load symptoms: {str(e)}")
            return []

    async def _load_medical_conditions(self) -> List[Dict]:
        """Load medical conditions/diseases from database"""
        try:
            query = """
            SELECT disease_id, name, description
            FROM diseases
            WHERE name IS NOT NULL
            ORDER BY name
            """
            async with self.db.get_session() as session:
                result = await session.execute(text(query))
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to load conditions: {str(e)}")
            return []

    async def initialize(self):
        """Initialize search engine components asynchronously"""
        try:
            # Load medical entities
            self.specialties = await self._load_medical_specialties()
            self.symptoms = await self._load_medical_symptoms()
            self.conditions = await self._load_medical_conditions()
            logger.info("Medical entities loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {str(e)}")
            raise

    def search(
        self,
        query: str,
        search_type: str = "doctors",
        filters: Optional[Dict] = None,
        location: Optional[str] = None,
        radius_km: Optional[float] = None,
        limit: int = 10
    ) -> Dict:
        """
        Execute semantic search across medical entities
        """
        try:
            # Validate input
            is_valid, error = self.validator.validate_query(query)
            if not is_valid:
                logger.error(f"Invalid query: {error}")
                return {
                    "results": [],
                    "total_count": 0,
                    "search_type": search_type,
                    "error": error
                }

            # Generate query embedding
            query_embedding = self.model.encode(query)

            # Prepare search parameters
            search_params = {
                "query_embedding": query_embedding.tolist(),
                "filters": filters or {},
                "location": location,
                "radius_km": radius_km,
                "limit": min(limit, 100)  # Cap at 100 results
            }

            # Execute search based on type
            if search_type == "doctors":
                results = self._search_doctors(**search_params)
            elif search_type == "hospitals":
                results = self._search_hospitals(**search_params)
            else:
                error = f"Invalid search_type: {search_type}"
                logger.error(error)
                return {
                    "results": [],
                    "total_count": 0,
                    "search_type": search_type,
                    "error": error
                }

            # Ensure results is a list
            results = results if results else []

            return {
                "results": results,
                "total_count": len(results),
                "search_type": search_type,
                "filters_applied": filters or {}
            }

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {
                "results": [],
                "total_count": 0,
                "search_type": search_type,
                "error": str(e)
            }

    def _search_doctors(
        self,
        query_embedding: List[float],
        filters: Dict,
        location: Optional[str],
        radius_km: Optional[float],
        limit: int
    ) -> List[Dict]:
        """Execute doctor-specific search"""
        try:
            query = """
            SELECT 
                d.did,
                d.name,
                d.specialization,
                d.city,
                d.experience,
                d.rating,
                d.consultant_fees_in_rupees,
                d.doctor_hours,
                1 - (d.embedding <-> :query_embedding::vector(384)) as similarity
            FROM doctors d
            WHERE TRUE
            """
            params = {"query_embedding": query_embedding}

            # Apply filters
            if filters.get('city'):
                query += " AND LOWER(d.city) = LOWER(:city)"
                params['city'] = filters['city']
                
            if filters.get('specialization'):
                query += " AND LOWER(d.specialization) = LOWER(:specialization)"
                params['specialization'] = filters['specialization']

            query += """
            ORDER BY similarity DESC
            LIMIT :limit
            """
            params['limit'] = limit

            return self.db.fetch_all(query, params)

        except Exception as e:
            logger.error(f"Doctor search failed: {str(e)}")
            return []

    def _search_hospitals(
        self,
        query_embedding: List[float],
        filters: Dict,
        location: Optional[str],
        radius_km: Optional[float],
        limit: int
    ) -> List[Dict]:
        """Execute hospital-specific search"""
        try:
            query = """
            SELECT 
                h.*,
                (embedding <=> %s) as distance
            FROM hospitals h
            WHERE TRUE
            """
            params = [query_embedding]

            # Apply filters
            if filters.get('hospital_type'):
                query += " AND hospital_type = %s"
                params.append(filters['hospital_type'])
            if filters.get('location'):
                query += " AND location = %s"
                params.append(filters['location'])

            query += """
            ORDER BY distance
            LIMIT %s
            """
            params.append(limit)

            results = self.db.fetch_all(query, params)
            return results if results else []

        except Exception as e:
            logger.error(f"Hospital search failed: {str(e)}")
            return []

    async def search_entity(self, entity_type: str, query: str, filters: Optional[Dict] = None, limit: int = 10) -> List[Dict]:
        """Execute search for specific entity type"""
        try:
            # Process query text
            processed_query = self.text_processor.process(query)
            
            # Generate query embedding
            query_embedding = self.model.encode(processed_query)

            # Build search query based on entity type
            base_query = f"""
            SELECT *,
            1 - (embedding <-> :embedding::vector(384)) as similarity_score
            FROM {entity_type}
            WHERE embedding IS NOT NULL
            AND (1 - (embedding <-> :embedding::vector(384))) > 0.1
            """
            
            params = {"embedding": query_embedding.tolist()}

            # Add filters if provided
            if filters:
                if entity_type == 'doctors':
                    if filters.get('city'):
                        base_query += " AND LOWER(city) = LOWER(:city)"
                        params['city'] = filters['city']
                    if filters.get('specialization'):
                        base_query += " AND (LOWER(specialization) = LOWER(:spec) OR LOWER(specialization) LIKE LOWER(:spec_pattern))"
                        params['spec'] = filters['specialization']
                        params['spec_pattern'] = f"%{filters['specialization']}%"
                elif entity_type in ['hospitals', 'clinics']:
                    if filters.get('city'):
                        base_query += " AND LOWER(location) = LOWER(:location)"
                        params['location'] = filters['city']

            # Add ordering and limit
            base_query += """
            ORDER BY similarity_score DESC, 
                     CASE WHEN LOWER(name) LIKE LOWER(:name_pattern) THEN 1 ELSE 0 END DESC
            LIMIT :limit
            """
            params.update({
                'name_pattern': f"%{query}%",
                'limit': limit
            })

            # Execute query
            async with self.db.get_session() as session:
                result = await session.execute(text(base_query), params)
                return [dict(row) for row in result.mappings()]

        except Exception as e:
            logger.error(f"Search failed for {entity_type}: {str(e)}")
            logger.error("Full traceback:", exc_info=True)
            return []

    def _fetch_results(self,
                      entity_type: str,
                      result_ids: List[str],
                      filters: Optional[Dict] = None) -> List[Dict]:
        """Fetch full results from database"""
        try:
            if not result_ids:
                return []

            query = f"""
                SELECT *
                FROM {entity_type}
                WHERE id = ANY(%s)
            """
            params = [result_ids]

            if filters:
                where_clauses = []
                for key, value in filters.items():
                    where_clauses.append(f"{key} = %s")
                    params.append(value)
                
                if where_clauses:
                    query += " AND " + " AND ".join(where_clauses)

            return self.db.execute_query(query, params)

        except Exception as e:
            logger.error(f"Failed to fetch results: {str(e)}")
            return []

    def _log_search_metrics(self,
                          query: str,
                          processed_query: str,
                          entity_types: List[str],
                          total_results: int,
                          execution_time: float,
                          expanded_terms: List[str]) -> None:
        """Log search metrics and patterns"""
        try:
            timestamp = datetime.now(timezone.utc)

            # Log search metrics
            self.db.execute("""
                INSERT INTO search_metrics 
                (query, entity_types, total_results, execution_time, result_types)
                VALUES (%s, %s, %s, %s, %s)
            """, [query, entity_types, total_results, execution_time, 
                 json.dumps({'entities': entity_types})])

            # Log query expansion
            if expanded_terms:
                self.db.execute("""
                    INSERT INTO query_expansion_logs
                    (original_query, expanded_terms, terms_added)
                    VALUES (%s, %s, %s)
                """, [query, json.dumps(expanded_terms), len(expanded_terms)])

            # Update search stats
            self.db.execute("""
                INSERT INTO search_stats (query, total_searches, last_searched_at)
                VALUES (%s, 1, %s)
                ON CONFLICT (query) DO UPDATE
                SET total_searches = search_stats.total_searches + 1,
                    last_searched_at = EXCLUDED.last_searched_at
            """, [query, timestamp])

        except Exception as e:
            logger.error(f"Failed to log search metrics: {str(e)}")