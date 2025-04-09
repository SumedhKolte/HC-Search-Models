class AnalysisQueries:
    @staticmethod
    def get_search_performance():
        return """
        SELECT 
            search_type,
            COUNT(*) as total_searches,
            AVG(results_count) as avg_results,
            AVG(execution_time) as avg_execution_time,
            SUM(CASE WHEN results_count > 0 THEN 1 ELSE 0 END)::float / COUNT(*) as success_rate
        FROM search_logs
        GROUP BY search_type
        ORDER BY total_searches DESC;
        """

    @staticmethod
    def get_popular_searches():
        return """
        SELECT 
            query,
            COUNT(*) as search_count,
            AVG(results_count) as avg_results,
            MAX(processed_query) as processed_query
        FROM search_logs
        GROUP BY query
        HAVING COUNT(*) > 5
        ORDER BY search_count DESC
        LIMIT 20;
        """

    @staticmethod
    def get_entity_coverage():
        return """
        SELECT 
            'doctors' as entity_type,
            COUNT(*) as total_count,
            COUNT(embedding) as embedded_count,
            COUNT(search_vector) as indexed_count
        FROM doctors
        UNION ALL
        SELECT 
            'hospitals',
            COUNT(*),
            COUNT(embedding),
            COUNT(search_vector)
        FROM hospitals
        UNION ALL
        SELECT 
            'clinics',
            COUNT(*),
            COUNT(embedding),
            COUNT(search_vector)
        FROM clinics;
        """