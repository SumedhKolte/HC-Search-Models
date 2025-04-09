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
        filters: Optional[Dict] = None
    ) -> Dict:
        """Build vector similarity search query"""
        
        # Get correct ID column name based on entity type
        id_column = f"{entity_type[:-1]}_id"  # converts 'doctors' to 'doctor_id'
        
        # Base query with vector similarity using correct column names
        query = f"""
        SELECT 
            t.{id_column},
            t.name,
            t.specialization,
            t.city,
            t.rating,
            t.experience,
            1 - (t.embedding <-> %(embedding)s) as similarity_score
        FROM {entity_type} t
        WHERE t.embedding IS NOT NULL
        AND (1 - (t.embedding <-> %(embedding)s)) > 0.01
        """

        # Add filters if provided
        params = {'embedding': query_embedding}
        if filters:
            for key, value in filters.items():
                if value and key != 'limit':
                    query += f" AND t.{key} = %({key})s"
                    params[key] = value

        # Add ordering and limit
        query += """
        ORDER BY similarity_score DESC
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
        """Build combined vector and text search query"""
        base_query = ""
        params = []

        if entity_type == 'doctors':
            base_query = """
            SELECT 
                did, name, specialization, city, experience, rating,
                consultant_fees_in_rupees, doctor_hours,
                1 - (embedding <-> %s::vector(384)) as similarity
            FROM doctors
            WHERE TRUE
            AND search_vector @@ plainto_tsquery(%s)
            """
            params = [query_embedding, text_query]

            if filters:
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
            base_query = """
            SELECT 
                hid, name, hospital_type, location, rating,
                1 - (embedding <-> %s::vector(384)) as similarity
            FROM hospitals
            WHERE TRUE
            AND search_vector @@ plainto_tsquery(%s)
            """
            params = [query_embedding, text_query]

            if filters:
                if filters.get('location'):
                    base_query += " AND LOWER(location) = LOWER(%s)"
                    params.append(filters['location'])
                if filters.get('hospital_type'):
                    base_query += " AND LOWER(hospital_type) = LOWER(%s)"
                    params.append(filters['hospital_type'])

        elif entity_type == 'diseases':
            base_query = """
            SELECT 
                disease_id, name, description, symptoms, treatments,
                specialty, severity, common_names,
                1 - (embedding <-> %s::vector(384)) as similarity
            FROM diseases
            WHERE TRUE
            AND search_vector @@ plainto_tsquery(%s)
            """
            params = [query_embedding, text_query]

        elif entity_type == 'symptoms':
            base_query = """
            SELECT 
                symptom_id, name, description, tags,
                1 - (embedding <-> %s::vector(384)) as similarity
            FROM symptoms
            WHERE TRUE
            AND search_vector @@ plainto_tsquery(%s)
            """
            params = [query_embedding, text_query]

        # Add common filter for limit
        base_query += " ORDER BY similarity DESC LIMIT %s"
        params.append(filters.get('limit', 10))

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