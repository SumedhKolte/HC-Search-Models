# from typing import Dict, List, Optional
# import json
# from datetime import datetime, timezone
# from src.data.queries import SearchQueries, MetricsQueries

# class SearchQueries:
#     """SQL queries for search operations"""
    
#     @staticmethod
#     def doctor_search(filters: Optional[Dict] = None) -> tuple:
#         """Generate doctor search query with optional filters"""
#         query = """
#         SELECT 
#             d.*,
#             h.name as hospital_name,
#             c.name as clinic_name
#         FROM doctors d
#         LEFT JOIN hospitals h ON d.hid = h.hid
#         LEFT JOIN clinics c ON d.cid = c.cid
#         WHERE 1=1
#         """
#         params = []
        
#         if filters:
#             if 'city' in filters:
#                 query += " AND d.city = %s"
#                 params.append(filters['city'])
#             if 'specialization' in filters:
#                 query += " AND d.specialization = %s"
#                 params.append(filters['specialization'])
#             if 'rating_min' in filters:
#                 query += " AND CAST(d.rating AS FLOAT) >= %s"
#                 params.append(float(filters['rating_min']))
                
#         return query, params

#     @staticmethod
#     def hospital_search(filters: Optional[Dict] = None) -> tuple:
#         """Generate hospital search query with optional filters"""
#         query = """
#         SELECT h.*,
#                COUNT(d.did) as doctor_count
#         FROM hospitals h
#         LEFT JOIN doctors d ON h.hid = d.hid
#         WHERE 1=1
#         """
#         params = []
        
#         if filters:
#             if 'location' in filters:
#                 query += " AND h.location = %s"
#                 params.append(filters['location'])
#             if 'hospital_type' in filters:
#                 query += " AND h.hospital_type = %s"
#                 params.append(filters['hospital_type'])
                
#         query += " GROUP BY h.hid"
#         return query, params

# class MetricsQueries:
#     """SQL queries for metrics and logging"""
    
#     @staticmethod
#     def log_search(
#         query: str,
#         search_type: str,
#         filters: Dict,
#         results_count: int,
#         user_agent: str,
#         ip_address: str,
#         processed_query: str
#     ) -> tuple:
#         """Generate query to log search attempt"""
#         query = """
#         INSERT INTO search_logs (
#             query, search_type, filters, results_count,
#             user_agent, ip_address, processed_query,
#             search_vector, created_at
#         ) VALUES (
#             %s, %s, %s, %s, %s, %s, %s,
#             to_tsvector('english', %s),
#             %s
#         )
#         """
#         params = [
#             query, search_type, json.dumps(filters), results_count,
#             user_agent, ip_address, processed_query,
#             processed_query, datetime.now(timezone.utc)
#         ]
#         return query, params

#     @staticmethod
#     def update_search_stats(query: str) -> str:
#         """Generate query to update search statistics"""
#         return """
#         INSERT INTO search_stats (query, total_searches, last_searched_at)
#         VALUES (%s, 1, %s)
#         ON CONFLICT (query) DO UPDATE
#         SET total_searches = search_stats.total_searches + 1,
#             last_searched_at = EXCLUDED.last_searched_at
#         """

# class ModelQueries:
#     """SQL queries for model training and evaluation"""
    
#     @staticmethod
#     def get_training_data(entity_type: str) -> str:
#         """Get training data for model"""
#         return f"""
#         SELECT *
#         FROM {entity_type}
#         WHERE embedding IS NULL
#         OR updated_at > last_embedded_at
#         LIMIT 1000
#         """

#     @staticmethod
#     def update_embeddings(entity_type: str) -> str:
#         """Update embeddings for entities"""
#         return f"""
#         UPDATE {entity_type}
#         SET 
#             embedding = %s,
#             last_embedded_at = %s
#         WHERE id = %s
#         """

# class AnalyticsQueries:
#     """SQL queries for analytics and reporting"""
    
#     @staticmethod
#     def get_search_patterns(days: int = 30) -> str:
#         """Get search patterns for analysis"""
#         return """
#         SELECT 
#             query,
#             search_type,
#             COUNT(*) as search_count,
#             AVG(results_count) as avg_results,
#             AVG(CASE WHEN results_count > 0 THEN 1 ELSE 0 END) as success_rate
#         FROM search_logs
#         WHERE created_at >= NOW() - INTERVAL '%s days'
#         GROUP BY query, search_type
#         HAVING COUNT(*) > 5
#         ORDER BY search_count DESC
#         """

#     @staticmethod
#     def get_performance_metrics(days: int = 7) -> str:
#         """Get system performance metrics"""
#         return """
#         SELECT 
#             DATE_TRUNC('hour', created_at) as time_bucket,
#             search_type,
#             COUNT(*) as total_searches,
#             AVG(execution_time) as avg_execution_time,
#             percentile_cont(0.95) WITHIN GROUP (ORDER BY execution_time) as p95_latency
#         FROM search_metrics
#         WHERE created_at >= NOW() - INTERVAL '%s days'
#         GROUP BY DATE_TRUNC('hour', created_at), search_type
#         ORDER BY time_bucket DESC
#         """

# class MaintenanceQueries:
#     """SQL queries for system maintenance"""
    
#     @staticmethod
#     def cleanup_old_logs(days: int = 90) -> str:
#         """Clean up old log entries"""
#         return """
#         DELETE FROM search_logs
#         WHERE created_at < NOW() - INTERVAL '%s days'
#         """

#     @staticmethod
#     def update_search_vectors() -> str:
#         """Update search vectors for all entities"""
#         return """
#         UPDATE doctors
#         SET search_vector = to_tsvector('english',
#             COALESCE(name, '') || ' ' ||
#             COALESCE(specialization, '') || ' ' ||
#             COALESCE(city, '') || ' ' ||
#             COALESCE(degree, '')
#         )
#         WHERE search_vector IS NULL
#         OR updated_at > last_vector_update;

#         UPDATE hospitals
#         SET search_vector = to_tsvector('english',
#             COALESCE(name, '') || ' ' ||
#             COALESCE(hospital_type, '') || ' ' ||
#             COALESCE(location, '')
#         )
#         WHERE search_vector IS NULL
#         OR updated_at > last_vector_update;
#         """

# # Example usage
# from src.data.database import Database  # Import the Database class

# db = Database()
# query, params = SearchQueries.doctor_search(filters={'city': 'delhi'})
# results = db.execute_query(query, params)

# # Log search
# query, params = MetricsQueries.log_search(
#     query="cardiologist in delhi",
#     search_type="doctor",
#     filters={"city": "delhi"},
#     results_count=10,
#     user_agent="Mozilla/5.0",
#     ip_address="127.0.0.1",
#     processed_query="cardiologist delhi"
# )
# db.execute(query, params)