import unittest
from pathlib import Path
import time
from datetime import datetime, timezone
import json
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from src.core.engine import SearchEngine
from src.data.database import Database
from src.search.vector import VectorSearch
from src.search.text import TextSearch
from src.search.location import LocationSearch
from src.utils.metrics import PerformanceMetrics

class TestPerformance(unittest.TestCase):
    """Performance and load testing for medical search system"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment and SSL certificate"""
        cls.setup_ssl()
        cls.db = Database()
        cls.engine = SearchEngine()
        cls.metrics = PerformanceMetrics()
        cls.vector_search = VectorSearch()
        cls.text_search = TextSearch()
        cls.location_search = LocationSearch()
        
        # Test queries
        cls.test_queries = [
            ("cardiologist in delhi", "doctors"),
            ("pediatrician near me", "doctors"),
            ("flu symptoms fever", "symptoms"),
            ("diabetes treatment", "diseases"),
            ("general hospital", "hospitals")
        ]
        
    @classmethod
    def setup_ssl(cls):
        """Setup SSL certificate for database connection"""
        ssl_dir = Path(__file__).parents[2] / 'ssl'
        ssl_dir.mkdir(exist_ok=True)
        cert_path = ssl_dir / 'rds-ca-bundle.pem'
        timestamp_path = ssl_dir / 'cert_timestamp.txt'
        
        needs_update = True
        if cert_path.exists() and timestamp_path.exists():
            with open(timestamp_path, 'r') as f:
                last_update = datetime.fromisoformat(f.read().strip())
                if (datetime.now(timezone.utc) - last_update).days < 30:
                    needs_update = False
        
        if needs_update:
            print(f"Downloading SSL certificate at {datetime.now(timezone.utc).isoformat()}")
            response = requests.get('https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem')
            response.raise_for_status()
            
            with open(cert_path, 'wb') as f:
                f.write(response.content)
            
            with open(timestamp_path, 'w') as f:
                f.write(datetime.now(timezone.utc).isoformat())
        
        return str(cert_path)

    def test_search_latency(self):
        """Test search response times"""
        latencies = []
        
        for query, entity_type in self.test_queries:
            start_time = time.time()
            results = self.engine.search(query=query, entity_type=entity_type)
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Assert maximum latency
            self.assertLess(latency, 2.0, f"Search latency too high: {latency:.2f}s")
            
            # Assert results returned
            self.assertIsInstance(results, list)
            
        # Log average latency
        avg_latency = np.mean(latencies)
        print(f"Average search latency: {avg_latency:.2f}s")
        
        self.metrics.log_search_metrics(
            query="performance_test",
            search_type="latency_test",
            results_count=len(self.test_queries),
            execution_time=avg_latency
        )

    def test_concurrent_searches(self):
        """Test system performance under concurrent load"""
        num_concurrent = 10
        results = []
        
        def execute_search(query_tuple):
            query, entity_type = query_tuple
            try:
                start_time = time.time()
                search_results = self.engine.search(query=query, entity_type=entity_type)
                latency = time.time() - start_time
                return {
                    'query': query,
                    'success': True,
                    'results_count': len(search_results),
                    'latency': latency
                }
            except Exception as e:
                return {
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'latency': 0
                }
        
        # Execute concurrent searches
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = []
            for _ in range(num_concurrent):
                for query_tuple in self.test_queries:
                    futures.append(executor.submit(execute_search, query_tuple))
            
            for future in futures:
                results.append(future.result())
        
        # Analyze results
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_latency = np.mean([r['latency'] for r in results if r['success']])
        
        print(f"Concurrent test results:")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average latency: {avg_latency:.2f}s")
        
        # Assert performance metrics
        self.assertGreater(success_rate, 0.95, "Too many failed requests")
        self.assertLess(avg_latency, 5.0, "Average latency too high under load")

    def test_vector_search_performance(self):
        """Test vector search performance"""
        # Test queries with different dimensions
        test_cases = [
            ("cardiologist experienced in heart surgery", 10),
            ("pediatrician with nicu experience", 20),
            ("orthopedic surgeon spine specialist", 30)
        ]
        
        for query, k in test_cases:
            start_time = time.time()
            results = self.vector_search.search(
                query=query,
                entity_type='doctors',
                k=k
            )
            latency = time.time() - start_time
            
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), k)
            self.assertLess(latency, 1.0, f"Vector search too slow: {latency:.2f}s")

    def test_location_search_scaling(self):
        """Test location-based search scaling"""
        test_cases = [
            ("delhi", 5.0),
            ("mumbai", 10.0),
            ("bangalore", 15.0)
        ]
        
        for location, radius in test_cases:
            start_time = time.time()
            results = self.location_search.search_nearby(
                location=location,
                entity_type='doctors',
                radius_km=radius
            )
            latency = time.time() - start_time
            
            self.assertIsInstance(results, list)
            self.assertLess(latency, 2.0, f"Location search too slow: {latency:.2f}s")

    def test_database_connection_pool(self):
        """Test database connection pool performance"""
        num_connections = 10
        queries = []
        
        for i in range(num_connections):
            queries.append(f"""
                SELECT COUNT(*) 
                FROM doctors 
                WHERE city = 'delhi'
            """)
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_connections) as executor:
            futures = [executor.submit(self.db.execute_query, query) for query in queries]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        avg_time = total_time / num_connections
        
        self.assertLess(avg_time, 0.5, f"Database connection pool too slow: {avg_time:.2f}s per query")

    def tearDown(self):
        """Clean up after tests"""
        # Log performance metrics
        self.db.execute("""
            INSERT INTO system_updates (status, error_message)
            VALUES (%s, %s)
        """, [
            True,
            json.dumps({
                'test_type': 'performance',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'completed'
            })
        ])

if __name__ == '__main__':
    unittest.main()