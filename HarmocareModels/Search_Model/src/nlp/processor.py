from typing import List, Optional, Dict
import re
import spacy
import logging
from datetime import datetime, timezone
import json
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker

from src.data.database import Database
from src.utils.logger import setup_logger
from src.utils.constants import EMBEDDING_MODELS

logger = setup_logger(__name__)

class TextProcessor:
    """Process and normalize text for medical search"""
    
    def __init__(self):
        """Initialize text processor with required models and resources"""
        self.db = Database()
        self.nlp = spacy.load('en_core_web_sm')
        self.spell = SpellChecker()
        
        # Common medical abbreviations
        self.abbreviations = {
            'dr': 'doctor',
            'dr.': 'doctor',
            'hosp': 'hospital',
            'hosp.': 'hospital',
            'dept': 'department',
            'dept.': 'department',
            'med': 'medical',
            'med.': 'medical'
        }
        
        # Load medical terms from database
        self.medical_terms = self._load_medical_terms()
        
    def process(self, text: str) -> str:
        """Process and normalize text"""
        try:
            if not text:
                return ""
                
            # Convert to lowercase
            text = text.lower().strip()
            
            # Expand abbreviations
            text = self._expand_abbreviations(text)
            
            # Autocorrect text
            text = self.autocorrect(text)
            
            # Process with spaCy
            doc = self.nlp(text)
            
            # Tokenize and normalize
            tokens = []
            for token in doc:
                # Skip punctuation and stopwords
                if token.is_punct or token.is_space:
                    continue
                    
                # Keep medical terms as is
                if token.text in self.medical_terms:
                    tokens.append(token.text)
                # Lemmatize other terms
                else:
                    tokens.append(token.lemma_)
            
            # Join tokens
            processed_text = ' '.join(tokens)
            
            # Log processing
            self._log_processing(original_text=text, processed_text=processed_text)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            return text

    def autocorrect(self, text: str) -> str:
        """Autocorrect text while preserving medical terms"""
        words = text.split()
        corrected = []
        
        for word in words:
            # Skip correction for medical terms
            if word.lower() in self.medical_terms:
                corrected.append(word)
                continue
                
            # Correct other words
            if not self.spell.correction(word) == word:
                correction = self.spell.correction(word)
                if correction:
                    corrected.append(correction)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
                
        return ' '.join(corrected)

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations"""
        words = text.split()
        expanded = []
        
        for word in words:
            expanded.append(self.abbreviations.get(word.lower(), word))
            
        return ' '.join(expanded)

    def _load_medical_terms(self) -> set:
        """Load medical terms from database"""
        try:
            # Get disease names and common names
            disease_query = """
            SELECT name, common_names
            FROM diseases
            """
            diseases = self.db.execute_query(disease_query)
            
            # Get symptom names
            symptom_query = """
            SELECT name, tags
            FROM symptoms
            """
            symptoms = self.db.execute_query(symptom_query)
            
            # Get specializations
            specialization_query = """
            SELECT DISTINCT specialization
            FROM doctors
            WHERE specialization IS NOT NULL
            """
            specializations = self.db.execute_query(specialization_query)
            
            # Combine all terms
            terms = set()
            
            for disease in diseases:
                terms.add(disease['name'].lower())
                if disease['common_names']:
                    terms.update([name.lower() for name in disease['common_names']])
                    
            for symptom in symptoms:
                terms.add(symptom['name'].lower())
                if symptom['tags']:
                    terms.update([tag.lower() for tag in symptom['tags']])
                    
            for spec in specializations:
                if spec['specialization']:
                    terms.add(spec['specialization'].lower())
            
            return terms
            
        except Exception as e:
            logger.error(f"Failed to load medical terms: {str(e)}")
            return set()

    def _log_processing(self, original_text: str, processed_text: str) -> None:
        """Log text processing details"""
        try:
            # Log to search_logs
            self.db.execute("""
                INSERT INTO search_logs 
                (query, processed_query, search_type, filters)
                VALUES (%s, %s, 'processing', %s)
            """, [
                original_text,
                processed_text,
                json.dumps({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'changes': len(processed_text) - len(original_text)
                })
            ])
            
        except Exception as e:
            logger.error(f"Failed to log text processing: {str(e)}")

class QueryProcessor:
    """Process and expand search queries"""
    
    def __init__(self):
        """Initialize query processor with embedding model"""
        self.model = SentenceTransformer(EMBEDDING_MODELS['bge_small'])
        
    async def process_query(self, query: str) -> Dict:
        """Process raw query text"""
        try:
            # Basic cleaning
            cleaned_query = query.strip().lower()
            
            # Generate embedding
            embedding = self.model.encode(cleaned_query, convert_to_numpy=True)
            
            return {
                'original_query': query,
                'processed_query': cleaned_query,
                'embedding': embedding.tolist(),
                'vector_dim': len(embedding)
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise
            
    async def expand_query(self, query: str) -> Dict:
        """Expand query with medical terms"""
        try:
            # Process original query
            query_data = await self.process_query(query)
            
            # TODO: Implement query expansion logic
            # This could include:
            # - Medical synonym lookup
            # - Common abbreviation expansion
            # - Related term suggestion
            expanded_terms = []
            
            return {
                **query_data,
                'expanded_terms': expanded_terms,
                'terms_added': len(expanded_terms)
            }
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            raise

    async def log_query(self, 
                       query_data: Dict,
                       results_count: int,
                       execution_time: float,
                       client_info: Optional[Dict] = None) -> None:
        """Log query metrics"""
        try:
            # TODO: Implement query logging
            # This should store:
            # - Original and processed queries
            # - Result metrics
            # - Client information
            # - Timestamp
            pass
            
        except Exception as e:
            logger.error(f"Query logging failed: {str(e)}")
            # Don't raise - logging failure shouldn't break search
            pass