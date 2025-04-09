from typing import List, Dict, Set, Optional
from datetime import datetime, timezone
import logging
import json
import spacy
from spacy.tokens import Doc
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.database import Database
from src.utils.logger import setup_logger
from config.settings import MODEL_CONFIG

logger = setup_logger(__name__)

class QueryExpander:
    """Expand medical search queries with relevant terms"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize query expander"""
        self.db = Database()
        self.nlp = spacy.load('en_core_web_sm')
        self.model = SentenceTransformer(MODEL_CONFIG['base_model'])
        self.similarity_threshold = similarity_threshold
        self.cache = {}
        
    def expand(self, query: str) -> List[str]:
        """Expand query with relevant medical terms"""
        try:
            start_time = datetime.now(timezone.utc)
            
            # Check cache first
            if query in self.cache:
                return self.cache[query]
            
            # Process query
            doc = self.nlp(query.lower())
            
            # Get potential terms to expand
            expansion_terms = set()
            
            # Add medical term expansions
            medical_terms = self._get_medical_terms(doc)
            if medical_terms:
                expansion_terms.update(medical_terms)
            
            # Add synonym expansions
            synonyms = self._get_synonyms(doc)
            if synonyms:
                expansion_terms.update(synonyms)
            
            # Add related disease/symptom terms
            related_terms = self._get_related_terms(doc)
            if related_terms:
                expansion_terms.update(related_terms)
            
            # Remove original query terms
            original_terms = {token.text for token in doc}
            expansion_terms = expansion_terms - original_terms
            
            # Convert to list and limit size
            expanded_terms = list(expansion_terms)[:5]  # Limit to top 5 expansions
            
            # Log expansion
            self._log_expansion(
                original_query=query,
                expanded_terms=expanded_terms,
                execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
            )
            
            # Update cache
            self.cache[query] = expanded_terms
            
            return expanded_terms
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return []

    def _get_medical_terms(self, doc: Doc) -> Set[str]:
        """Get relevant medical terms from diseases and symptoms"""
        try:
            # Extract potential medical terms from query
            medical_terms = set()
            
            for token in doc:
                if token.pos_ in ['NOUN', 'ADJ']:
                    # Query diseases
                    query = """
                    SELECT name, common_names
                    FROM diseases
                    WHERE name ILIKE %s 
                    OR %s = ANY(common_names)
                    """
                    results = self.db.execute_query(query, [f"%{token.text}%", token.text])
                    
                    for result in results:
                        medical_terms.add(result['name'])
                        if result['common_names']:
                            medical_terms.update(result['common_names'])
                    
                    # Query symptoms
                    query = """
                    SELECT name, tags
                    FROM symptoms
                    WHERE name ILIKE %s
                    OR %s = ANY(tags)
                    """
                    results = self.db.execute_query(query, [f"%{token.text}%", token.text])
                    
                    for result in results:
                        medical_terms.add(result['name'])
                        if result['tags']:
                            medical_terms.update(result['tags'])
            
            return medical_terms
            
        except Exception as e:
            logger.error(f"Failed to get medical terms: {str(e)}")
            return set()

    def _get_synonyms(self, doc: Doc) -> Set[str]:
        """Get synonyms for query terms"""
        try:
            synonyms = set()
            
            for token in doc:
                if token.pos_ in ['NOUN', 'ADJ', 'VERB']:
                    # Get similar terms based on embeddings
                    query_embedding = self.model.encode(token.text)
                    
                    # Query similar terms from search logs
                    query = """
                    SELECT processed_query
                    FROM search_logs
                    WHERE results_count > 0
                    AND search_vector @@ to_tsquery('english', %s)
                    LIMIT 10
                    """
                    results = self.db.execute_query(query, [token.text])
                    
                    if results:
                        terms = [r['processed_query'] for r in results]
                        term_embeddings = self.model.encode(terms)
                        
                        # Calculate similarities
                        similarities = np.dot(term_embeddings, query_embedding) / \
                                    (np.linalg.norm(term_embeddings, axis=1) * np.linalg.norm(query_embedding))
                        
                        # Add similar terms
                        for term, sim in zip(terms, similarities):
                            if sim > self.similarity_threshold:
                                synonyms.add(term)
            
            return synonyms
            
        except Exception as e:
            logger.error(f"Failed to get synonyms: {str(e)}")
            return set()

    def _get_related_terms(self, doc: Doc) -> Set[str]:
        """Get related medical terms"""
        try:
            related_terms = set()
            
            # Get disease-symptom relationships
            for token in doc:
                if token.pos_ == 'NOUN':
                    # Query related diseases
                    query = """
                    SELECT d.name, d.common_names
                    FROM diseases d
                    JOIN symptoms s ON d.disease_id = ANY(s.related_diseases)
                    WHERE s.name ILIKE %s
                    """
                    results = self.db.execute_query(query, [f"%{token.text}%"])
                    
                    for result in results:
                        related_terms.add(result['name'])
                        if result['common_names']:
                            related_terms.update(result['common_names'])
            
            return related_terms
            
        except Exception as e:
            logger.error(f"Failed to get related terms: {str(e)}")
            return set()

    def _log_expansion(self, 
                      original_query: str,
                      expanded_terms: List[str],
                      execution_time: float) -> None:
        """Log query expansion details"""
        try:
            self.db.execute("""
                INSERT INTO query_expansion_logs
                (original_query, expanded_terms, terms_added, execution_time)
                VALUES (%s, %s, %s, %s)
            """, [
                original_query,
                json.dumps(expanded_terms),
                len(expanded_terms),
                execution_time
            ])
            
        except Exception as e:
            logger.error(f"Failed to log query expansion: {str(e)}")