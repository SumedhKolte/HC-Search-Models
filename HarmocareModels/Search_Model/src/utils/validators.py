from typing import Any, Dict, List, Optional, Tuple
import re
from datetime import datetime, timezone
import uuid
import logging
from pathlib import Path
import requests

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class InputValidator:
    """Validate input data for medical search system"""
    
    def __init__(self):
        """Initialize validator with SSL certificate check"""
        self._verify_ssl_cert()
        
        # Common regex patterns
        self.email_pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
        self.phone_pattern = re.compile(r'^\+?[0-9]{10,15}$')
        self.uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        
    def _verify_ssl_cert(self) -> None:
        """Verify SSL certificate"""
        try:
            ssl_dir = Path(__file__).parents[3] / 'ssl'
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
                logger.info(f"Downloading SSL certificate at {datetime.now(timezone.utc).isoformat()}")
                response = requests.get('https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem')
                response.raise_for_status()
                
                with open(cert_path, 'wb') as f:
                    f.write(response.content)
                
                with open(timestamp_path, 'w') as f:
                    f.write(datetime.now(timezone.utc).isoformat())
                    
        except Exception as e:
            logger.error(f"SSL certificate verification failed: {str(e)}")
            raise

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate search query"""
        if not query:
            return False, "Query cannot be empty"
        
        if len(query) > 500:
            return False, "Query too long (max 500 characters)"
            
        # Check for basic SQL injection patterns
        if any(pattern in query.lower() for pattern in ['--', ';', 'union', 'select']):
            return False, "Invalid query characters"
            
        return True, ""

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """Validate geographic coordinates"""
        try:
            if not (-90 <= float(lat) <= 90):
                return False
            if not (-180 <= float(lon) <= 180):
                return False
            return True
        except (ValueError, TypeError):
            return False

    def validate_doctor(self, doctor_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate doctor data"""
        errors = []
        
        # Required fields
        required = ['name', 'gender', 'city', 'specialization']
        for field in required:
            if field not in doctor_data or not doctor_data[field]:
                errors.append(f"Missing required field: {field}")
                
        # Email validation
        if doctor_data.get('email'):
            if not self.email_pattern.match(doctor_data['email']):
                errors.append("Invalid email format")
                
        # Phone validation
        if doctor_data.get('phone'):
            if not self.phone_pattern.match(doctor_data['phone']):
                errors.append("Invalid phone format")
                
        # UUID validation
        for field in ['hid', 'cid']:
            if doctor_data.get(field):
                if not self.uuid_pattern.match(str(doctor_data[field])):
                    errors.append(f"Invalid {field} format")
        
        return len(errors) == 0, errors

    def validate_hospital(self, hospital_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate hospital data"""
        errors = []
        
        # Required fields
        required = ['name', 'hospital_type', 'location']
        for field in required:
            if field not in hospital_data or not hospital_data[field]:
                errors.append(f"Missing required field: {field}")
                
        # Rating validation
        if hospital_data.get('rating'):
            try:
                rating = float(hospital_data['rating'])
                if not (0 <= rating <= 5):
                    errors.append("Rating must be between 0 and 5")
            except ValueError:
                errors.append("Invalid rating format")
        
        return len(errors) == 0, errors

    def validate_filters(self, filters: Dict[str, Any], entity_type: str) -> Tuple[bool, List[str]]:
        """Validate search filters"""
        errors = []
        
        allowed_filters = {
            'doctors': ['city', 'specialization', 'gender', 'rating'],
            'hospitals': ['hospital_type', 'rating', 'location'],
            'clinics': ['location'],
            'diseases': ['tags'],
            'symptoms': ['related_diseases']
        }
        
        if entity_type not in allowed_filters:
            return False, ["Invalid entity type"]
            
        for key in filters:
            if key not in allowed_filters[entity_type]:
                errors.append(f"Invalid filter: {key}")
                
        return len(errors) == 0, errors

    def validate_embedding(self, embedding: Any) -> bool:
        """Validate embedding format"""
        try:
            if not isinstance(embedding, (bytes, bytearray)):
                return False
                
            # Check embedding dimension
            import numpy as np
            vector = np.frombuffer(embedding, dtype=np.float32)
            return len(vector) in [384, 512, 768]  # Common embedding dimensions
            
        except Exception:
            return False

    def sanitize_query(self, query: str) -> str:
        """Sanitize search query"""
        # Remove special characters
        query = re.sub(r'[^\w\s\-\.]', '', query)
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        return query.lower()