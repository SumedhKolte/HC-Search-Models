from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data.database import Database
from src.utils.metrics import PerformanceMetrics
import logging
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)

class SearchQueryBuilder:
    def __init__(self):
        self.db = Database()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.metrics = PerformanceMetrics()

    def build_vector_query(
        self,
        entity_type: str,
        query_embedding: List[float],
        filters: Optional[Dict] = None,
        search_text: Optional[str] = None  # Add search_text parameter
    ) -> Dict:
        """Build vector similarity search query with enhanced name matching"""
        
        # Get correct ID column name based on entity type
        id_column = f"{entity_type[:-1]}_id"  
        
        # Base query with combined scoring
        query = f"""
        SELECT 
            t.{id_column},
            t.name,
            t.specialization,
            t.city,
            t.rating,
            t.experience,
            GREATEST(
                1 - (t.embedding <-> %(embedding)s),
                CASE 
                    WHEN %(search_text)s IS NOT NULL AND 
                         LOWER(t.name) LIKE LOWER(%(name_pattern)s) THEN 1.0
                    WHEN %(search_text)s IS NOT NULL AND 
                         STRING_TO_ARRAY(LOWER(t.name), ' ') && 
                         STRING_TO_ARRAY(LOWER(%(search_text)s), ' ') THEN 0.8
                    ELSE 1 - (t.embedding <-> %(embedding)s)
                END
            ) as similarity_score
        FROM {entity_type} t
        WHERE t.embedding IS NOT NULL
        AND (
            (1 - (t.embedding <-> %(embedding)s)) > 0.01
            OR (%(search_text)s IS NOT NULL AND 
                (LOWER(t.name) LIKE LOWER(%(name_pattern)s) 
                 OR STRING_TO_ARRAY(LOWER(t.name), ' ') && 
                    STRING_TO_ARRAY(LOWER(%(search_text)s), ' ')))
        )
        """

        # Set up parameters including name search
        params = {
            'embedding': query_embedding,
            'search_text': search_text,
            'name_pattern': f'%{search_text}%' if search_text else None
        }

        # Add filters if provided
        if filters:
            for key, value in filters.items():
                if value and key != 'limit':
                    query += f" AND LOWER(t.{key}) = LOWER(%({key})s)"
                    params[key] = value

        # Add ordering and limit
        query += """
        ORDER BY 
            similarity_score DESC,
            CASE 
                WHEN %(search_text)s IS NOT NULL AND 
                     LOWER(name) LIKE LOWER(%(name_pattern)s) THEN 0 
                ELSE 1 
            END,
            t.rating DESC NULLS LAST
        LIMIT %(limit)s
        """
        params['limit'] = filters.get('limit', 10) if filters else 10

        logger.debug(f"Generated query: {query}")
        logger.debug(f"Query params: {params}")

        return {
            'query': query,
            'params': params
        }

    def build_keyword_query(self, entity_type: str, keywords: str, filters: Dict) -> tuple:
        """Build keyword search query"""
        base_query = f"""
        SELECT *,
            ts_rank_cd(search_vector, plainto_tsquery(%s)) as rank
        FROM {entity_type}
        WHERE search_vector @@ plainto_tsquery(%s)
        """
        params = [keywords, keywords]

        if filters:
            for key, value in filters.items():
                if value:
                    base_query += f" AND LOWER({key}) = LOWER(%s)"
                    params.append(value)

        base_query += " ORDER BY rank DESC LIMIT %s"
        params.append(filters.get('limit', 10))

        return base_query, params

    @staticmethod
    def build_combined_search_query(query_embedding: List[float], text_query: str,
                                  entity_type: str, filters: Dict = None) -> tuple:
        """Build combined vector and text search query with name matching"""
        base_query = ""
        params = []

        if entity_type == 'doctors':
            # Split the name into parts for more flexible matching
            name_parts = text_query.lower().split()
            
            base_query = """
            SELECT 
                did, name, specialization, city, experience, rating,
                consultantfee, doctor_hours,
                GREATEST(
                    1 - (embedding <-> %s::vector(384)),
                    ts_rank_cd(search_vector, plainto_tsquery(%s)),
                    CASE 
                        WHEN LOWER(name) LIKE ALL(SELECT '%' || LOWER(unnest(%s)) || '%') THEN 1.0
                        WHEN LOWER(name) LIKE ANY(SELECT '%' || LOWER(unnest(%s)) || '%') THEN 0.8
                        ELSE 0.0
                    END
                ) as similarity
            FROM doctors
            WHERE TRUE
            AND (
                search_vector @@ plainto_tsquery(%s)
                OR LOWER(name) LIKE ANY(SELECT '%' || LOWER(unnest(%s)) || '%')
                OR EXISTS (
                    SELECT 1 FROM diseases d
                    WHERE d.name ILIKE %s
                    AND d.disease_id = ANY(doctors.specializations)
                )
                OR EXISTS (
                    SELECT 1 FROM symptoms s
                    WHERE s.name ILIKE %s
                    AND s.symptom_id = ANY(doctors.conditions_treated)
                )
            )
            """
            params = [
                query_embedding,          # Vector similarity
                text_query,              # Full text search
                name_parts,              # Array for exact name match (all parts)
                name_parts,              # Array for partial name match (any part)
                text_query,              # Full text search condition
                name_parts,              # Array for name LIKE condition
                f"%{text_query}%",       # Disease name match
                f"%{text_query}%"        # Symptom name match
            ]

        elif entity_type == 'hospitals':
            base_query = """
            SELECT 
                hid, name, hospital_type, location, rating,
                GREATEST(
                    1 - (embedding <-> %s::vector(384)),
                    ts_rank_cd(search_vector, plainto_tsquery(%s)),
                    CASE 
                        WHEN LOWER(name) LIKE LOWER(%s) THEN 1.0
                        WHEN LOWER(name) LIKE LOWER(%s) THEN 0.8
                        ELSE 0.0
                    END
                ) as similarity
            FROM hospitals
            WHERE TRUE
            AND (
                search_vector @@ plainto_tsquery(%s)
                OR LOWER(name) LIKE LOWER(%s)
                OR LOWER(name) LIKE LOWER(%s)
                OR EXISTS (
                    SELECT 1 FROM diseases d
                    WHERE d.name ILIKE %s
                    AND d.disease_id = ANY(hospitals.treated_conditions)
                )
            )
            """
            name_pattern = f"%{text_query}%"
            word_pattern = f"% {text_query} %"
            params = [
                query_embedding,      # Vector similarity
                text_query,          # Full text search
                name_pattern,        # Exact name match
                word_pattern,        # Partial name match
                text_query,          # Full text search condition
                name_pattern,        # Name LIKE condition
                word_pattern,        # Word in name condition
                f"%{text_query}%"    # Disease name match
            ]

        elif entity_type == 'clinics':
            base_query = """
            SELECT 
                cid, name, location, specialties,
                GREATEST(
                    1 - (embedding <-> %s::vector(384)),
                    ts_rank_cd(search_vector, plainto_tsquery(%s)),
                    CASE 
                        WHEN LOWER(name) LIKE LOWER(%s) THEN 1.0
                        WHEN LOWER(name) LIKE LOWER(%s) THEN 0.8
                        ELSE 0.0
                    END
                ) as similarity
            FROM clinics
            WHERE TRUE
            AND (
                search_vector @@ plainto_tsquery(%s)
                OR LOWER(name) LIKE LOWER(%s)
                OR LOWER(name) LIKE LOWER(%s)
                OR EXISTS (
                    SELECT 1 FROM diseases d
                    WHERE d.name ILIKE %s
                    AND d.disease_id = ANY(clinics.treated_conditions)
                )
            )
            """
            name_pattern = f"%{text_query}%"
            word_pattern = f"% {text_query} %"
            params = [
                query_embedding,      # Vector similarity
                text_query,          # Full text search
                name_pattern,        # Exact name match 
                word_pattern,        # Partial name match
                text_query,          # Full text search condition
                name_pattern,        # Name LIKE condition
                word_pattern,        # Word in name condition
                f"%{text_query}%"    # Disease name match
            ]

        # Apply common filters
        if filters:
            if entity_type == 'doctors':
                if filters.get('city'):
                    base_query += " AND LOWER(city) = LOWER(%s)"
                    params.append(filters['city'])
                if filters.get('specialization'):
                    base_query += " AND LOWER(specialization) = LOWER(%s)"
                    params.append(filters['specialization'])
                if filters.get('min_rating'):
                    base_query += " AND CAST(rating AS FLOAT) >= %s"
                    params.append(float(filters['min_rating']))

            elif entity_type == 'hospitals':
                if filters.get('location'):
                    base_query += " AND LOWER(location) = LOWER(%s)"
                    params.append(filters['location'])
                if filters.get('hospital_type'):
                    base_query += " AND LOWER(hospital_type) = LOWER(%s)"
                    params.append(filters['hospital_type'])

        # Add ORDER BY and LIMIT
        base_query += """
        ORDER BY 
            similarity DESC,
            CASE WHEN LOWER(name) LIKE LOWER(%s) THEN 0 ELSE 1 END,
            rating DESC NULLS LAST
        LIMIT %s
        """
        params.extend([f"%{text_query}%", filters.get('limit', 10)])

        return base_query, params

    @staticmethod
    def build_symptom_based_search() -> str:
        """Build query for symptom-based doctor/disease search"""
        return """
        WITH matching_diseases AS (
            SELECT d.disease_id, d.name as disease_name
            FROM diseases d
            JOIN symptoms s ON s.symptom_id = ANY(d.related_symptoms)
            WHERE LOWER(s.name) = LOWER(%s)
        )
        SELECT 
            doc.did,
            doc.name as doctor_name,
            doc.specialization,
            doc.city,
            doc.rating,
            md.disease_name as treats_disease
        FROM doctors doc
        JOIN matching_diseases md ON 
            doc.specializations && ARRAY[md.disease_id]::uuid[]
        WHERE TRUE
        """

    def build_vector_search_query(self, entity_type: str, 
                                query_embedding: List[float], 
                                filters: Dict = None) -> Tuple[str, Dict]:
        """Build vector similarity search query"""
        sql = f"""
        SELECT 
            *,
            1 - (embedding <-> %(query_embedding)s::vector(384)) as similarity
        FROM {entity_type}
        WHERE TRUE
        """
        params = {'query_embedding': query_embedding}

        if filters:
            if entity_type == 'doctors':
                if filters.get('city'):
                    sql += " AND LOWER(city) = LOWER(%(city)s)"
                    params['city'] = filters['city']
                if filters.get('specialization'):
                    sql += " AND LOWER(specialization) = LOWER(%(specialization)s)"
                    params['specialization'] = filters['specialization']

        sql += " ORDER BY similarity DESC LIMIT %(limit)s"
        params['limit'] = filters.get('limit', 10)

        return sql, params