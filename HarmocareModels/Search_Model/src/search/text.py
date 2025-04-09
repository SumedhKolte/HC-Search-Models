from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timezone
import json

from src.data.database import Database
from src.utils.logger import setup_logger
from src.nlp.processor import TextProcessor
from src.nlp.expander import QueryExpander

logger = setup_logger(__name__)

class TextSearch:
    """Text-based search implementation using PostgreSQL full-text search"""
    
    def __init__(self):
        """Initialize text search"""
        self.db = Database()
        self.text_processor = TextProcessor()
        self.query_expander = QueryExpander()
        
    def search(self,
              query: str,
              entity_type: str,
              filters: Optional[Dict] = None,
              limit: int = 10,
              expand_query: bool = True) -> List[Dict]:
        """Perform text search on specified entity type"""
        try:
            start_time = datetime.now(timezone.utc)
            
            # Process query
            processed_query = self.text_processor.process(query)
            
            # Expand query if requested
            expanded_terms = []
            if expand_query:
                expanded_terms = self.query_expander.expand(processed_query)
                processed_query = f"{processed_query} {' '.join(expanded_terms)}"
            
            # Build search query based on entity type
            if entity_type == 'doctors':
                results = self._search_doctors(processed_query, filters, limit)
            elif entity_type == 'hospitals':
                results = self._search_hospitals(processed_query, filters, limit)
            elif entity_type == 'clinics':
                results = self._search_clinics(processed_query, filters, limit)
            elif entity_type == 'diseases':
                results = self._search_diseases(processed_query, filters, limit)
            elif entity_type == 'symptoms':
                results = self._search_symptoms(processed_query, filters, limit)
            else:
                raise ValueError(f"Unsupported entity type: {entity_type}")
            
            # Log search metrics
            self._log_search_metrics(
                query=query,
                processed_query=processed_query,
                entity_type=entity_type,
                results_count=len(results),
                execution_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
                expanded_terms=expanded_terms
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Text search failed: {str(e)}")
            return []

    def _search_doctors(self,
                       query: str,
                       filters: Optional[Dict],
                       limit: int) -> List[Dict]:
        """Search doctors table"""
        try:
            base_query = """
            SELECT 
                d.*,
                h.name as hospital_name,
                c.name as clinic_name,
                ts_rank_cd(d.search_vector, to_tsquery('english', %s)) as rank
            FROM doctors d
            LEFT JOIN hospitals h ON d.hid = h.hid
            LEFT JOIN clinics c ON d.cid = c.cid
            WHERE d.search_vector @@ to_tsquery('english', %s)
            """
            
            params = [query, query]
            
            if filters:
                for key, value in filters.items():
                    base_query += f" AND d.{key} = %s"
                    params.append(value)
            
            base_query += " ORDER BY rank DESC LIMIT %s"
            params.append(limit)
            
            return self.db.execute_query(base_query, params)
            
        except Exception as e:
            logger.error(f"Doctor search failed: {str(e)}")
            return []

    def _search_hospitals(self,
                         query: str,
                         filters: Optional[Dict],
                         limit: int) -> List[Dict]:
        """Search hospitals table"""
        try:
            base_query = """
            SELECT 
                h.*,
                ts_rank_cd(h.search_vector, to_tsquery('english', %s)) as rank
            FROM hospitals h
            WHERE h.search_vector @@ to_tsquery('english', %s)
            """
            
            params = [query, query]
            
            if filters:
                for key, value in filters.items():
                    base_query += f" AND h.{key} = %s"
                    params.append(value)
            
            base_query += " ORDER BY rank DESC LIMIT %s"
            params.append(limit)
            
            return self.db.execute_query(base_query, params)
            
        except Exception as e:
            logger.error(f"Hospital search failed: {str(e)}")
            return []

    def _search_diseases(self,
                        query: str,
                        filters: Optional[Dict],
                        limit: int) -> List[Dict]:
        """Search diseases table"""
        try:
            base_query = """
            SELECT 
                d.*,
                ts_rank_cd(d.search_vector, to_tsquery('english', %s)) as rank
            FROM diseases d
            WHERE d.search_vector @@ to_tsquery('english', %s)
            OR %s = ANY(d.common_names)
            """
            
            params = [query, query, query.lower()]
            
            if filters:
                for key, value in filters.items():
                    base_query += f" AND d.{key} = %s"
                    params.append(value)
            
            base_query += " ORDER BY rank DESC LIMIT %s"
            params.append(limit)
            
            return self.db.execute_query(base_query, params)
            
        except Exception as e:
            logger.error(f"Disease search failed: {str(e)}")
            return []

    def _search_symptoms(self,
                        query: str,
                        filters: Optional[Dict],
                        limit: int) -> List[Dict]:
        """Search symptoms table"""
        try:
            base_query = """
            SELECT 
                s.*,
                d.name as disease_name,
                ts_rank_cd(s.search_vector, to_tsquery('english', %s)) as rank
            FROM symptoms s
            LEFT JOIN diseases d ON d.disease_id = ANY(s.related_diseases)
            WHERE s.search_vector @@ to_tsquery('english', %s)
            """
            
            params = [query, query]
            
            if filters:
                for key, value in filters.items():
                    base_query += f" AND s.{key} = %s"
                    params.append(value)
            
            base_query += " ORDER BY rank DESC LIMIT %s"
            params.append(limit)
            
            return self.db.execute_query(base_query, params)
            
        except Exception as e:
            logger.error(f"Symptom search failed: {str(e)}")
            return []

    def _search_clinics(self,
                       query: str,
                       filters: Optional[Dict],
                       limit: int) -> List[Dict]:
        """Search clinics table"""
        try:
            base_query = """
            SELECT 
                c.*,
                ts_rank_cd(c.search_vector, to_tsquery('english', %s)) as rank
            FROM clinics c
            WHERE c.search_vector @@ to_tsquery('english', %s)
            """
            
            params = [query, query]
            
            if filters:
                for key, value in filters.items():
                    base_query += f" AND c.{key} = %s"
                    params.append(value)
            
            base_query += " ORDER BY rank DESC LIMIT %s"
            params.append(limit)
            
            return self.db.execute_query(base_query, params)
            
        except Exception as e:
            logger.error(f"Clinic search failed: {str(e)}")
            return []

    def _log_search_metrics(self,
                          query: str,
                          processed_query: str,
                          entity_type: str,
                          results_count: int,
                          execution_time: float,
                          expanded_terms: List[str]) -> None:
        """Log search metrics"""
        try:
            # Log to search_metrics
            self.db.execute("""
                INSERT INTO search_metrics
                (query, total_results, execution_time, result_types)
                VALUES (%s, %s, %s, %s)
            """, [
                query,
                results_count,
                execution_time,
                json.dumps({
                    'type': 'text_search',
                    'entity_type': entity_type
                })
            ])
            
            # Log to search_logs
            self.db.execute("""
                INSERT INTO search_logs
                (query, processed_query, search_type, results_count, filters)
                VALUES (%s, %s, %s, %s, %s)
            """, [
                query,
                processed_query,
                entity_type,
                results_count,
                json.dumps({
                    'expanded_terms': expanded_terms,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            ])
            
        except Exception as e:
            logger.error(f"Failed to log search metrics: {str(e)}")