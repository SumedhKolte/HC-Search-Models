from src.data.database import Database
from src.data.Analysis.analysis_queries import AnalysisQueries
from src.data.Analysis.model_queries import ModelQueries
import pandas as pd

class AnalyticsQueries:
    @staticmethod
    def get_search_trends():
        return """
        SELECT 
            DATE_TRUNC('day', TO_TIMESTAMP(log_id)) as search_date,
            search_type,
            COUNT(*) as search_count,
            AVG(results_count) as avg_results
        FROM search_logs
        GROUP BY DATE_TRUNC('day', TO_TIMESTAMP(log_id)), search_type
        ORDER BY search_date DESC
        LIMIT 30;
        """

    @staticmethod
    def get_recommendation_effectiveness():
        return """
        SELECT 
            query_type,
            city,
            AVG(results_count) as avg_results,
            AVG(execution_time) as avg_execution_time,
            COUNT(*) as total_recommendations
        FROM recommendtion_metrics
        GROUP BY query_type, city
        HAVING COUNT(*) > 10
        ORDER BY avg_results DESC;
        """

def run_analysis():
    db = Database()
    
    # Get search performance metrics
    search_perf = pd.read_sql_query(
        AnalysisQueries.get_search_performance(),
        db.engine
    )
    
    # Get model performance metrics
    model_perf = pd.read_sql_query(
        ModelQueries.get_model_performance(),
        db.engine
    )
    
    # Generate reports
    search_perf.to_csv('reports/search_performance.csv')
    model_perf.to_csv('reports/model_performance.csv')

if __name__ == "__main__":
    run_analysis()