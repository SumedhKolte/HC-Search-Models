class ModelQueries:
    @staticmethod
    def get_model_performance():
        return """
        SELECT 
            entity_type,
            AVG(execution_time) as avg_execution_time,
            MIN(execution_time) as min_execution_time,
            MAX(execution_time) as max_execution_time,
            SUM(CASE WHEN success THEN 1 ELSE 0 END)::float / COUNT(*) as success_rate
        FROM model_traing_metrics
        GROUP BY entity_type;
        """

    @staticmethod
    def get_query_expansion_stats():
        return """
        SELECT 
            terms_added,
            COUNT(*) as query_count,
            AVG(array_length(expanded_terms::text[], 1)) as avg_terms
        FROM query_expansion_logs
        GROUP BY terms_added
        ORDER BY terms_added;
        """