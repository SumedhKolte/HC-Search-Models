import unittest
from pathlib import Path
import json
from datetime import datetime, timezone
import requests
import numpy as np

from src.nlp.processor import TextProcessor
from src.nlp.expander import QueryExpander
from src.nlp.analyzer import QueryAnalyzer
from src.data.database import Database

class TestNLP(unittest.TestCase):
    """Unit tests for NLP components"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment and SSL certificate"""
        cls.setup_ssl()
        cls.db = Database()
        cls.processor = TextProcessor()
        cls.expander = QueryExpander()
        cls.analyzer = QueryAnalyzer()
        
        # Test data
        cls.test_queries = [
            "cardiologist in delhi",
            "fever and cough symptoms",
            "pediatrician near hospital",
            "Dr. Smith appointment"
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

    def test_text_processor(self):
        """Test text processing functionality"""
        # Test basic processing
        text = "Dr. Smith is a Cardiologist in Delhi Hospital"
        processed = self.processor.process(text)
        self.assertIsInstance(processed, str)
        self.assertTrue(processed.islower())
        
        # Test abbreviation expansion
        text = "dept. of med."
        processed = self.processor.process(text)
        self.assertIn("department", processed)
        self.assertIn("medical", processed)
        
        # Test empty input
        self.assertEqual(self.processor.process(""), "")
        
        # Test special characters
        text = "heart-disease & symptoms"
        processed = self.processor.process(text)
        self.assertNotIn("&", processed)
        
        # Verify medical terms preserved
        text = "myocardial infarction symptoms"
        processed = self.processor.process(text)
        self.assertIn("myocardial", processed)
        self.assertIn("infarction", processed)

    def test_query_expansion(self):
        """Test query expansion functionality"""
        # Test basic expansion
        query = "flu symptoms"
        expanded = self.expander.expand(query)
        self.assertIsInstance(expanded, list)
        self.assertGreater(len(expanded), 0)
        
        # Test with medical terms
        query = "cardiologist"
        expanded = self.expander.expand(query)
        self.assertTrue(any("heart" in term for term in expanded))
        
        # Test with location
        query = "doctor in delhi"
        expanded = self.expander.expand(query)
        self.assertFalse(any(term.startswith("in ") for term in expanded))
        
        # Test expansion limit
        query = "general physician consultation"
        expanded = self.expander.expand(query, max_terms=3)
        self.assertLessEqual(len(expanded), 3)
        
        # Verify expansion logging
        self.db.execute_query("""
            SELECT * FROM query_expansion_logs
            WHERE original_query = %s
            ORDER BY log_id DESC
            LIMIT 1
        """, [query])

    def test_query_analyzer(self):
        """Test query analysis functionality"""
        # Test entity detection
        query = "cardiologist in delhi"
        analysis = self.analyzer.analyze(query)
        self.assertIn('entity_type', analysis)
        self.assertEqual(analysis['entity_type'], 'doctors')
        
        # Test location extraction
        self.assertIn('location', analysis)
        self.assertEqual(analysis['location'], 'delhi')
        
        # Test specialization detection
        self.assertIn('specialization', analysis)
        self.assertEqual(analysis['specialization'], 'cardiology')
        
        # Test symptom query
        query = "fever and cough"
        analysis = self.analyzer.analyze(query)
        self.assertEqual(analysis['entity_type'], 'symptoms')
        
        # Test disease query
        query = "diabetes treatment"
        analysis = self.analyzer.analyze(query)
        self.assertEqual(analysis['entity_type'], 'diseases')

    def test_analyzer_edge_cases(self):
        """Test query analyzer edge cases"""
        # Empty query
        analysis = self.analyzer.analyze("")
        self.assertIsNone(analysis.get('entity_type'))
        
        # Invalid location
        analysis = self.analyzer.analyze("doctor in xyz123")
        self.assertIsNone(analysis.get('location'))
        
        # Multiple entities
        analysis = self.analyzer.analyze("cardiologist and neurologist")
        self.assertEqual(analysis['entity_type'], 'doctors')
        self.assertIn('specializations', analysis)
        
        # Ambiguous query
        analysis = self.analyzer.analyze("medical")
        self.assertIn('confidence', analysis)
        self.assertLess(analysis['confidence'], 1.0)

    def test_processor_performance(self):
        """Test text processor performance"""
        start_time = datetime.now(timezone.utc)
        
        for query in self.test_queries:
            processed = self.processor.process(query)
            self.assertIsNotNone(processed)
        
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Log metrics
        self.db.execute("""
            INSERT INTO model_training_metrics
            (entity_type, embeddings_count, dimension, executiontime, success)
            VALUES (%s, %s, %s, %s, %s)
        """, [
            'text_processor',
            len(self.test_queries),
            0,
            execution_time,
            True
        ])
        
        # Assert performance
        self.assertLess(execution_time / len(self.test_queries), 0.1)

if __name__ == '__main__':
    unittest.main()