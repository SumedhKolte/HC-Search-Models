import unittest
from pathlib import Path
import json
from datetime import datetime, timezone
import requests
from typing import Dict, List, Optional

from src.core.engine import SearchEngine
from src.data.database import Database
from src.nlp.processor import TextProcessor
from src.nlp.expander import QueryExpander
from src.search.vector import VectorSearch
from src.search.text import TextSearch
from src.search.location import LocationSearch
from src.search.ranking import ResultRanker

class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests for medical search system"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment and SSL certificate"""
        cls.setup_ssl()
        cls.db = Database()
        cls.engine = SearchEngine()
        cls.processor = TextProcessor()
        cls.expander = QueryExpander()
        cls.vector_search = VectorSearch()
        cls.text_search = TextSearch()
        cls.location_search = LocationSearch()
        cls.ranker = ResultRanker()
        
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

    def test_doctor_search_workflow(self):
        """Test complete doctor search workflow"""
        query = "cardiologist in delhi with 5 years experience"
        
        # Process query
        processed_query = self.processor.process(query)
        self.assertIsNotNone(processed_query)
        
        # Expand query
        expanded_terms = self.expander.expand(processed_query)
        self.assertIsInstance(expanded_terms, list)
        
        # Vector search
        vector_results = self.vector_search.search(
            query=processed_query,
            entity_type='doctors',
            filters={'city': 'delhi'}
        )
        self.assertIsInstance(vector_results, list)
        
        # Text search
        text_results = self.text_search.search(
            query=processed_query,
            entity_type='doctors',
            filters={'city': 'delhi'}
        )
        self.assertIsInstance(text_results, list)
        
        # Rank results
        final_results = self.ranker.rank_results(
            results=vector_results + text_results,
            entity_type='doctors',
            query_embedding=self.vector_search.model.encode(processed_query),
            query_text=processed_query
        )
        
        self.assertIsInstance(final_results, list)
        if final_results:
            self.assertIn('did', final_results[0])
            self.assertIn('name', final_results[0])
            self.assertIn('specialization', final_results[0])

    def test_disease_symptom_search(self):
        """Test disease and symptom search integration"""
        query = "flu symptoms with fever"
        
        # Search diseases
        disease_results = self.text_search.search(
            query=query,
            entity_type='diseases',
            expand_query=True
        )
        self.assertIsInstance(disease_results, list)
        
        # Search symptoms
        symptom_results = self.text_search.search(
            query=query,
            entity_type='symptoms',
            expand_query=True
        )
        self.assertIsInstance(symptom_results, list)
        
        if disease_results:
            self.assertIn('disease_id', disease_results[0])
            self.assertIn('name', disease_results[0])
        
        if symptom_results:
            self.assertIn('symptom_id', symptom_results[0])
            self.assertIn('name', symptom_results[0])

    def test_location_based_search(self):
        """Test location-based search workflow"""
        location = "delhi"
        radius_km = 5.0
        
        # Search nearby doctors
        doctor_results = self.location_search.search_nearby(
            location=location,
            entity_type='doctors',
            radius_km=radius_km,
            filters={'specialization': 'cardiology'}
        )
        self.assertIsInstance(doctor_results, list)
        
        # Search nearby hospitals
        hospital_results = self.location_search.search_nearby(
            location=location,
            entity_type='hospitals',
            radius_km=radius_km
        )
        self.assertIsInstance(hospital_results, list)
        
        if doctor_results:
            self.assertIn('did', doctor_results[0])
            self.assertIn('distance_km', doctor_results[0])
        
        if hospital_results:
            self.assertIn('hid', hospital_results[0])
            self.assertIn('distance_km', hospital_results[0])

    def test_search_logging(self):
        """Test search logging functionality"""
        query = "pediatrician"
        
        # Perform search
        results = self.engine.search(
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
        self.assertEqual(log_entry[0]['search_type'], 'doctors')
        self.assertIsNotNone(log_entry[0]['results_count'])

    def test_error_handling(self):
        """Test error handling and logging"""
        # Invalid query
        with self.assertRaises(Exception):
            self.engine.search(
                query="",  # Empty query
                entity_type='doctors'
            )
        
        # Invalid entity type
        with self.assertRaises(ValueError):
            self.engine.search(
                query="test",
                entity_type='invalid_type'
            )
        
        # Invalid filters
        with self.assertRaises(ValueError):
            self.engine.search(
                query="test",
                entity_type='doctors',
                filters={'invalid_field': 'value'}
            )

if __name__ == '__main__':
    unittest.main()