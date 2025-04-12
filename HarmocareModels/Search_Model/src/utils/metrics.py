from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime, timezone
import json
import logging
import asyncio
import statistics
from pathlib import Path
from sqlalchemy import text
from src.data.database import Database

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Track and analyze search system performance metrics"""
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics_dir = Path('monitoring/metrics')
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.metrics_dir / 'performance.log'
        self.start_time = None
        self.metrics = {}
        self._db = None

    @property
    def db(self):
        """Lazy load database to avoid circular import"""
        if self._db is None:
            # Import here to avoid circular dependency
            from src.data.database import get_db
            self._db = get_db()
        return self._db

    def start(self):
        """Start timing"""
        self.start_time = datetime.now(timezone.utc)

    def stop(self, results_count: int) -> Dict[str, Any]:
        """Stop timing and calculate metrics"""
        if not self.start_time:
            return {}
            
        end_time = datetime.now(timezone.utc)
        execution_time = (end_time - self.start_time).total_seconds() * 1000
        
        self.metrics = {
            'execution_time_ms': execution_time,
            'results_count': results_count,
            'timestamp': end_time.isoformat()
        }
        
        return self.metrics

    def log(self, message: str):
        """Log a performance metric message"""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_message = f"[METRIC][{timestamp}] {message}"
        print(log_message)
        
        # Also write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{log_message}\n")
        except Exception as e:
            logger.error(f"Failed to write to metrics log: {str(e)}")

    def log_search_metrics(
        self, 
        query: str,
        search_type: str,
        results_count: int,
        execution_time: float,
        filters: Dict = None
    ) -> None:
        """Log search metrics"""
        try:
            with self.db.SessionLocal() as session:
                session.execute(
                    text("""
                    INSERT INTO search_metrics (
                        query, search_type, results_count,
                        execution_time, filters, created_at
                    ) VALUES (:query, :search_type, :results_count,
                            :execution_time, :filters, :created_at)
                    """),
                    {
                        'query': query,
                        'search_type': search_type,
                        'results_count': results_count,
                        'execution_time': execution_time,
                        'filters': filters,
                        'created_at': datetime.now(timezone.utc)
                    }
                )
                session.commit()
        except Exception as e:
            logger.error(f"Failed to log search metrics: {str(e)}")

    def log_model_metrics(self, entity_type: str, embeddings_count: int,
                         dimension: int, execution_time: float,
                         success: bool, error_message: str = None) -> None:
        """Log model training metrics"""
        try:
            sql = """
            INSERT INTO model_training_metrics
                (entity_type, embeddings_count, dimension, execution_time, 
                 success, error_message, created_at)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s)
            """
            
            params = [
                entity_type,
                embeddings_count,
                dimension,
                execution_time,
                success,
                error_message,
                datetime.now(timezone.utc)
            ]
            
            self.db.execute(sql, params)
            
        except Exception as e:
            logger.error(f"Failed to log model metrics: {str(e)}")

    def calculate_search_performance(self,
                                   days: int = 30) -> Dict[str, Any]:
        """Calculate search performance metrics"""
        try:
            query = """
            WITH recent_searches AS (
                SELECT 
                    search_type,
                    results_count,
                    execution_time,
                    CASE WHEN results_count > 0 THEN 1 ELSE 0 END as success
                FROM search_metrics
                WHERE created_at >= NOW() - INTERVAL '%s days'
            )
            SELECT 
                search_type,
                COUNT(*) as total_searches,
                AVG(results_count) as avg_results,
                AVG(execution_time) as avg_execution_time,
                AVG(success) as success_rate
            FROM recent_searches
            GROUP BY search_type
            """
            
            results = self.db.execute_query(query, [days])
            
            metrics = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'period_days': days,
                'metrics': results
            }
            
            # Save metrics to file
            metrics_file = self.metrics_dir / f'search_performance_{datetime.now(timezone.utc).strftime("%Y%m%d")}.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate search performance: {str(e)}")
            return {}

    def get_recommendation_metrics(self,
                                 city: Optional[str] = None,
                                 days: int = 30) -> Dict[str, Any]:
        """Get recommendation system metrics"""
        try:
            query = """
            SELECT 
                query_type,
                city,
                AVG(results_count) as avg_results,
                AVG(execution_time) as avg_execution_time,
                COUNT(*) as total_recommendations
            FROM recommendation_metrics
            WHERE created_at >= NOW() - INTERVAL '%s days'
            """
            params = [days]
            
            if city:
                query += " AND city = %s"
                params.append(city)
                
            query += " GROUP BY query_type, city"
            
            results = self.db.execute_query(query, params)
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'period_days': days,
                'city': city,
                'metrics': results
            }
            
        except Exception as e:
            logger.error(f"Failed to get recommendation metrics: {str(e)}")
            return {}

    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns and expansion effectiveness"""
        try:
            query = """
            WITH query_stats AS (
                SELECT 
                    q.original_query,
                    q.terms_added,
                    s.results_count,
                    s.execution_time
                FROM query_expansion_logs q
                JOIN search_logs s ON q.original_query = s.query
            )
            SELECT 
                terms_added,
                COUNT(*) as query_count,
                AVG(results_count) as avg_results,
                AVG(execution_time) as avg_execution_time
            FROM query_stats
            GROUP BY terms_added
            ORDER BY terms_added
            """
            
            results = self.db.execute_query(query)
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'expansion_metrics': results
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze query patterns: {str(e)}")
            return {}

    def save_system_status(self, status: bool, message: str) -> None:
        """Save system update status"""
        try:
            self.db.execute("""
                INSERT INTO system_updates (status, error_message)
                VALUES (%s, %s)
            """, [status, message])
            
        except Exception as e:
            logger.error(f"Failed to save system status: {str(e)}")

    async def log_search(
        self,
        query: str,
        entity_types: Union[List[str], str],
        total_results: int = 0,
        filters: Dict = None,
        processed_query: str = None,
        execution_time: float = 0
    ) -> None:
        """Log search details with sanitized inputs"""
        try:
            # Normalize entity_types to always be a list
            if isinstance(entity_types, str):
                entity_types = [entity_types]

            insert_query = text("""
            INSERT INTO search_logs (
                query,
                search_type,
                results_count,
                entity_types,
                processed_query,
                filters,
                created_at,
                execution_time
            ) VALUES (
                :query,
                :search_type,
                :results_count,
                :entity_types,
                :processed_query,
                :filters,
                :created_at,
                :execution_time
            )
            """)
            
            params = {
                'query': query,
                'search_type': entity_types[0],  # Use first type as primary
                'results_count': total_results,
                'entity_types': entity_types,  # Pass list directly, let SQLAlchemy handle casting
                'processed_query': processed_query or query,
                'filters': json.dumps(filters) if filters else None,
                'created_at': datetime.now(timezone.utc),
                'execution_time': execution_time
            }

            async with self.db.get_session() as session:
                await session.execute(insert_query, params)
                await session.commit()
                logger.info(f"Search logged successfully: {query} with {total_results} results")

        except Exception as e:
            logger.error(f"Failed to log search: {str(e)}")
            logger.exception("Full traceback:")

def calculate_metrics(search_results: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """Calculate search performance metrics"""
    metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'mrr': 0.0,  # Mean Reciprocal Rank
        'latency_ms': 0.0
    }
    
    try:
        if not search_results or not ground_truth:
            return metrics
            
        # Calculate metrics
        relevant = set(r['id'] for r in ground_truth)
        retrieved = set(r['id'] for r in search_results)
        
        true_positives = len(relevant.intersection(retrieved))
        
        metrics['precision'] = true_positives / len(retrieved) if retrieved else 0
        metrics['recall'] = true_positives / len(relevant) if relevant else 0
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            
        # Calculate MRR
        for i, result in enumerate(search_results, 1):
            if result['id'] in relevant:
                metrics['mrr'] = 1.0 / i
                break
                
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {str(e)}")
        raise

# Use string for type hint to avoid circular import
def log_metrics(db: 'Database', metrics: Dict[str, float], query: str, search_type: str) -> None:
    """Log search metrics to database"""
    try:
        # Implementation...
        pass
    except Exception as e:
        logger.error(f"Failed to log metrics: {str(e)}")

def log_db_performance() -> Dict[str, Any]:
    """Log database performance metrics"""
    try:
        # Import here to avoid circular imports
        from src.data.database import Database

        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'queries': {},
            'connection_stats': {},
            'table_stats': {}
        }

        # Initialize database
        db = Database()

        # Test connection and get version
        with db.engine.connect() as conn:
            # Basic connection test
            start = datetime.now(timezone.utc)
            version = conn.execute(text("SELECT version()")).scalar()
            end = datetime.now(timezone.utc)
            
            metrics['connection_stats'] = {
                'latency_ms': (end - start).total_seconds() * 1000,
                'status': 'connected',
                'version': version
            }

            # Get table statistics
            table_stats = conn.execute(text("""
                SELECT 
                    schemaname,
                    relname as table_name,
                    n_live_tup as row_count,
                    pg_total_relation_size(relid) as total_bytes
                FROM pg_stat_user_tables
                WHERE schemaname = 'public'
            """)).mappings().all()

            metrics['table_stats'] = {
                row['table_name']: {
                    'rows': row['row_count'],
                    'size_mb': row['total_bytes'] / (1024 * 1024)
                }
                for row in table_stats
            }

            # Test common queries
            test_queries = {
                'doctor_search': "SELECT COUNT(*) FROM doctors WHERE specialization ILIKE '%cardiology%'",
                'vector_search': "SELECT COUNT(*) FROM doctors WHERE embedding IS NOT NULL",
                'text_search': "SELECT COUNT(*) FROM doctors WHERE search_vector @@ plainto_tsquery('heart specialist')",
                'join_performance': """
                    SELECT COUNT(*) 
                    FROM doctors d 
                    JOIN hospitals h ON d.hid = h.hid
                """
            }

            for name, query in test_queries.items():
                start_time = datetime.now(timezone.utc)
                result = conn.execute(text(query)).scalar()
                end_time = datetime.now(timezone.utc)
                
                metrics['queries'][name] = {
                    'execution_time_ms': (end_time - start_time).total_seconds() * 1000,
                    'result': result
                }

            # Calculate summary
            query_times = [q['execution_time_ms'] for q in metrics['queries'].values()]
            metrics['summary'] = {
                'avg_query_time_ms': statistics.mean(query_times),
                'total_tables': len(metrics['table_stats']),
                'total_rows': sum(t['rows'] for t in metrics['table_stats'].values()),
                'total_size_mb': sum(t['size_mb'] for t in metrics['table_stats'].values())
            }

            # Log metrics to database
            conn.execute(
                text("""
                    INSERT INTO system_metrics 
                    (metric_type, details, created_at) 
                    VALUES ('database_performance', :metrics, :timestamp)
                """),
                {
                    'metrics': json.dumps(metrics),
                    'timestamp': datetime.now(timezone.utc)
                }
            )

            logger.info(f"✅ Database performance logged: avg_query_time={metrics['summary']['avg_query_time_ms']:.2f}ms")
            return metrics

    except Exception as e:
        logger.error(f"❌ Failed to log database performance: {str(e)}")
        logger.exception("Full traceback:")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }