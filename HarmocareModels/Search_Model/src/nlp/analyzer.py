from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime, timezone
import logging
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import json

from src.data.database import Database
from src.utils.logger import setup_logger
from src.nlp.processor import TextProcessor

logger = setup_logger(__name__)

class QueryAnalyzer:
    """Analyze search queries and patterns"""
    
    def __init__(self):
        """Initialize query analyzer"""
        self.db = Database()
        self.nlp = spacy.load('en_core_web_sm')
        self.text_processor = TextProcessor()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze single query structure and intent"""
        try:
            doc = self.nlp(query.lower())
            
            # Extract entities and patterns
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Identify key components
            specializations = []
            locations = []
            symptoms = []
            
            for token in doc:
                if token.pos_ == 'NOUN':
                    # Check medical terms
                    if self._is_medical_term(token.text):
                        symptoms.append(token.text)
                    # Check specializations
                    elif self._is_specialization(token.text):
                        specializations.append(token.text)
                elif token.ent_type_ == 'GPE':
                    locations.append(token.text)
            
            analysis = {
                'query': query,
                'processed_query': self.text_processor.process(query),
                'entities': entities,
                'specializations': specializations,
                'locations': locations,
                'symptoms': symptoms,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Log analysis
            self._log_query_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            raise

    def analyze_search_patterns(self, 
                              days: int = 30,
                              min_searches: int = 5) -> Dict[str, Any]:
        """Analyze search patterns over time period"""
        try:
            # Get search patterns
            query = """
            SELECT 
                query,
                processed_query,
                search_type,
                COUNT(*) as search_count,
                AVG(results_count) as avg_results,
                AVG(CASE WHEN results_count > 0 THEN 1 ELSE 0 END) as success_rate
            FROM search_logs
            WHERE created_at >= NOW() - INTERVAL '%s days'
            GROUP BY query, processed_query, search_type
            HAVING COUNT(*) >= %s
            ORDER BY search_count DESC
            """
            
            patterns = self.db.execute_query(query, [days, min_searches])
            
            # Analyze common terms
            all_queries = ' '.join([p['processed_query'] for p in patterns])
            term_frequencies = Counter(all_queries.split())
            
            # Calculate success metrics
            success_metrics = {
                'total_searches': sum(p['search_count'] for p in patterns),
                'avg_success_rate': np.mean([p['success_rate'] for p in patterns]),
                'top_terms': dict(term_frequencies.most_common(10))
            }
            
            # Get query expansion stats
            expansion_stats = self._get_expansion_stats(days)
            
            analysis = {
                'time_period_days': days,
                'total_unique_queries': len(patterns),
                'patterns': patterns,
                'success_metrics': success_metrics,
                'expansion_stats': expansion_stats,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {str(e)}")
            raise

    def _is_medical_term(self, term: str) -> bool:
        """Check if term is medical"""
        query = """
        SELECT EXISTS(
            SELECT 1 
            FROM diseases 
            WHERE name ILIKE %s 
            OR %s = ANY(common_names)
            UNION
            SELECT 1 
            FROM symptoms 
            WHERE name ILIKE %s
        )
        """
        result = self.db.execute_query(query, [f"%{term}%", term, f"%{term}%"])
        return result[0]['exists'] if result else False

    def _is_specialization(self, term: str) -> bool:
        """Check if term is medical specialization"""
        query = """
        SELECT EXISTS(
            SELECT 1 
            FROM doctors 
            WHERE specialization ILIKE %s
        )
        """
        result = self.db.execute_query(query, [f"%{term}%"])
        return result[0]['exists'] if result else False

    def _get_expansion_stats(self, days: int) -> Dict[str, Any]:
        """Get query expansion statistics"""
        query = """
        SELECT 
            AVG(terms_added) as avg_terms_added,
            AVG(array_length(expanded_terms::text[], 1)) as avg_expansion_length,
            COUNT(*) as total_expansions
        FROM query_expansion_logs
        WHERE created_at >= NOW() - INTERVAL '%s days'
        """
        
        return self.db.execute_query(query, [days])[0]

    def _log_query_analysis(self, analysis: Dict[str, Any]) -> None:
        """Log query analysis results"""
        try:
            # Log to search_logs
            self.db.execute("""
                INSERT INTO search_logs 
                (query, processed_query, search_type, filters)
                VALUES (%s, %s, 'analysis', %s)
            """, [
                analysis['query'],
                analysis['processed_query'],
                json.dumps({
                    'entities': analysis['entities'],
                    'specializations': analysis['specializations'],
                    'locations': analysis['locations'],
                    'symptoms': analysis['symptoms']
                })
            ])
            
        except Exception as e:
            logger.error(f"Failed to log query analysis: {str(e)}")