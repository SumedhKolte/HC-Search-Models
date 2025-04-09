import unittest
from pathlib import Path
import json
from datetime import datetime, timezone
import requests
import numpy as np
from typing import Dict, List

from src.search.vector import VectorSearch
from src.search.text import TextSearch
from src.search.location import LocationSearch
from src.search.ranking import ResultRanker
from src.data.database import Database
from src.utils.validators import InputValidator

class TestSearch(unittest.TestCase):
    """Unit tests for search components"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment and SSL certificate"""
        cls.setup_ssl()
        cls.db = Database()
        cls.vector_search = VectorSearch()
        cls.text_search = TextSearch()
        cls.location_search = LocationSearch()
        cls.ranker = ResultRanker()
        cls.validator = InputValidator()
        
        # Test queries
        cls.test_queries = {
            'doctor': {
                'query': 'cardiologist in delhi',
                'filters': {'city': 'delhi'}
            },
            'hospital': {
                'query': 'cardiac hospital',
                'filters': {'hospital_type': 'general'}
            },
            'disease': {
                'query': 'flu symptoms',
                'filters': None
            }
        }
        
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

    def test_vector_search(self):
        """Test vector search functionality"""
        query = self.test_queries['doctor']['query']
        
        # Test basic search
        results = self.vector_search.search(
            query=query,
            entity_type='doctors',
            k=5
        )
        self.assertIsInstance(results, list)
        if results:
            self.assertIn('did', results[0])
            
        # Test with filters
        filtered_results = self.vector_search.search(
            query=query,
            entity_type='doctors',
            filters=self.test_queries['doctor']['filters'],
            k=5
        )
        if filtered_results:
            self.assertEqual(filtered_results[0]['city'].lower(), 'delhi')
            
        # Test similarity scores
        if results:
            self.assertIn('similarity_score', results[0])
            self.assertGreaterEqual(results[0]['similarity_score'], 0.0)
            self.assertLessEqual(results[0]['similarity_score'], 1.0)

    def test_text_search(self):
        """Test text search functionality"""
        query = self.test_queries['disease']['query']
        
        # Test basic search
        results = self.text_search.search(
            query=query,
            entity_type='diseases',
            expand_query=True
        )
        self.assertIsInstance(results, list)
        if results:
            self.assertIn('disease_id', results[0])
            
        # Test query expansion
        expanded_results = self.text_search.search(
            query=query,
            entity_type='diseases',
            expand_query=True
        )
        if expanded_results:
            self.assertGreaterEqual(len(expanded_results), len(results))

    def test_location_search(self):
        """Test location-based search functionality"""
        location = "delhi"
        radius_km = 5.0
        
        # Test nearby doctors
        results = self.location_search.search_nearby(
            location=location,
            entity_type='doctors',
            radius_km=radius_km,
            filters={'specialization': 'cardiology'}
        )
        self.assertIsInstance(results, list)
        if results:
            self.assertIn('distance_km', results[0])
            self.assertLessEqual(results[0]['distance_km'], radius_km)
            
        # Test coordinate validation
        self.assertTrue(self.validator.validate_coordinates(28.6139, 77.2090))  # Delhi coords
        self.assertFalse(self.validator.validate_coordinates(999, 999))

    def test_ranking(self):
        """Test result ranking functionality"""
        # Get sample results
        vector_results = self.vector_search.search(
            query=self.test_queries['doctor']['query'],
            entity_type='doctors',
            k=5
        )
        
        text_results = self.text_search.search(
            query=self.test_queries['doctor']['query'],
            entity_type='doctors'
        )
        
        combined_results = vector_results + text_results
        
        if combined_results:
            # Test ranking
            ranked_results = self.ranker.rank_results(
                results=combined_results,
                entity_type='doctors',
                query_embedding=self.vector_search.model.encode(
                    self.test_queries['doctor']['query']
                ),
                query_text=self.test_queries['doctor']['query']
            )
            
            self.assertIsInstance(ranked_results, list)
            self.assertEqual(len(ranked_results), len(combined_results))
            
            if len(ranked_results) > 1:
                # Verify ranking order
                self.assertGreaterEqual(
                    ranked_results[0]['scores']['total_score'],
                    ranked_results[-1]['scores']['total_score']
                )

    def test_search_logging(self):
        """Test search logging functionality"""
        query = self.test_queries['doctor']['query']
        
        # Execute search
        results = self.text_search.search(
            query=query,
            entity_type='doctors'
        )
        
        # Verify log entry
        log_entry = self.db.execute_query("""
            SELECT *
            FROM search_logs
            WHERE query = %s
            ORDER BY log_id DESC
            LIMIT 1
        """, [query])
        
        self.assertTrue(log_entry)
        self.assertEqual(log_entry[0]['query'], query)
        self.assertEqual(log_entry[0]['results_count'], len(results))

    def tearDown(self):
        """Clean up after tests"""
        # Log test execution
        self.db.execute("""
            INSERT INTO system_updates (status, error_message)
            VALUES (%s, %s)
        """, [
            True,
            json.dumps({
                'test_type': 'search_unit',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'completed'
            })
        ])

if __name__ == '__main__':
    unittest.main()