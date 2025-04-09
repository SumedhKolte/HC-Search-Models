# important for listing queries need to optimise this file for performance and readability

from typing import Dict, List, Optional
from sqlalchemy import text

class SearchQueries:
    @staticmethod
    def doctor_search(query_embedding: List[float], filters: Dict) -> tuple:
        """Generate doctor search query"""
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
            1 - (d.embedding <-> CAST(%(query_embedding)s AS vector(384))) as similarity
        FROM doctors d
        WHERE TRUE
        """
        params = {'query_embedding': query_embedding}

        if filters.get('city'):
            query += " AND LOWER(d.city) = LOWER(%(city)s)"
            params['city'] = filters['city']

        if filters.get('specialization'):
            query += " AND LOWER(d.specialization) = LOWER(%(specialization)s)"
            params['specialization'] = filters['specialization']

        query += """
        ORDER BY similarity DESC
        LIMIT %(limit)s
        """
        params['limit'] = filters.get('limit', 10)

        return query, params

    @staticmethod
    def hospital_search(query_embedding: List[float], filters: Dict) -> tuple:
        """Generate hospital search query"""
        query = """
        SELECT 
            h.hid,
            h.name,
            h.hospital_type,
            h.location,
            h.rating,
            h.embedding <-> %(query_embedding)s::vector as distance
        FROM hospitals h
        WHERE TRUE
        """
        params = {'query_embedding': query_embedding}

        if filters.get('location'):
            query += " AND LOWER(h.location) = LOWER(%(location)s)"
            params['location'] = filters['location']

        if filters.get('hospital_type'):
            query += " AND LOWER(h.hospital_type) = LOWER(%(type)s)"
            params['type'] = filters['hospital_type']

        query += " ORDER BY distance LIMIT %(limit)s"
        params['limit'] = filters.get('limit', 10)

        return text(query), params

    @staticmethod
    def disease_symptom_search(query_embedding: List[float], search_type: str, filters: Dict) -> tuple:
        """Search diseases or symptoms"""
        table = 'diseases' if search_type == 'diseases' else 'symptoms'
        query = f"""
        SELECT 
            {table}.*,
            {table}.embedding <-> %(query_embedding)s::vector as distance
        FROM {table}
        WHERE TRUE
        """
        params = {'query_embedding': query_embedding}

        if filters.get('text_query'):
            query += f" AND {table}.search_vector @@ plainto_tsquery(%(text_query)s)"
            params['text_query'] = filters['text_query']

        query += " ORDER BY distance LIMIT %(limit)s"
        params['limit'] = filters.get('limit', 10)

        return text(query), params