from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timezone
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import json

from src.data.database import Database
from src.utils.logger import setup_logger
from src.utils.validators import validate_coordinates

logger = setup_logger(__name__)

class LocationSearch:
    """Location-based search for medical entities"""
    
    def __init__(self):
        """Initialize location search"""
        self.db = Database()
        self.geocoder = Nominatim(
            user_agent="medical_search",
            timeout=10
        )
        self.location_cache = {}
        
    def search_nearby(self,
                     location: str,
                     entity_type: str,
                     radius_km: float = 5.0,
                     limit: int = 10,
                     filters: Optional[Dict] = None) -> List[Dict]:
        """Search for entities near a location"""
        try:
            start_time = datetime.now(timezone.utc)
            
            # Get coordinates for location
            coordinates = self._get_coordinates(location)
            if not coordinates:
                logger.error(f"Could not geocode location: {location}")
                return []
            
            lat, lon = coordinates
            
            # Build query based on entity type
            if entity_type == 'doctors':
                results = self._search_doctors_nearby(lat, lon, radius_km, limit, filters)
            elif entity_type == 'hospitals':
                results = self._search_hospitals_nearby(lat, lon, radius_km, limit, filters)
            elif entity_type == 'clinics':
                results = self._search_clinics_nearby(lat, lon, radius_km, limit, filters)
            else:
                raise ValueError(f"Unsupported entity type: {entity_type}")
            
            # Log search metrics
            self._log_location_search(
                location=location,
                entity_type=entity_type,
                radius_km=radius_km,
                results_count=len(results),
                execution_time=(datetime.now(timezone.utc) - start_time).total_seconds()
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Location search failed: {str(e)}")
            return []

    def _get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for location with caching"""
        try:
            # Check cache first
            if location in self.location_cache:
                return self.location_cache[location]
            
            # Geocode location
            location_data = self.geocoder.geocode(location)
            if not location_data:
                return None
                
            coordinates = (location_data.latitude, location_data.longitude)
            
            # Validate coordinates
            if not validate_coordinates(coordinates[0], coordinates[1]):
                return None
                
            # Cache result
            self.location_cache[location] = coordinates
            
            return coordinates
            
        except GeocoderTimedOut:
            logger.error(f"Geocoding timed out for location: {location}")
            return None
        except Exception as e:
            logger.error(f"Geocoding failed: {str(e)}")
            return None

    def _search_doctors_nearby(self,
                             lat: float,
                             lon: float,
                             radius_km: float,
                             limit: int,
                             filters: Optional[Dict] = None) -> List[Dict]:
        """Search for doctors near coordinates"""
        try:
            query = """
            WITH nearby_locations AS (
                SELECT 
                    d.*,
                    h.location as hospital_location,
                    c.location as clinic_location,
                    h.name as hospital_name,
                    c.name as clinic_name
                FROM doctors d
                LEFT JOIN hospitals h ON d.hid = h.hid
                LEFT JOIN clinics c ON d.cid = c.cid
                WHERE d.city IS NOT NULL
            )
            SELECT *
            FROM nearby_locations
            WHERE (
                ST_DWithin(
                    ST_MakePoint(%s, %s)::geography,
                    COALESCE(hospital_location, clinic_location)::geography,
                    %s * 1000  -- Convert km to meters
                )
            )
            """
            params = [lon, lat, radius_km]  # Note: ST_MakePoint takes lon, lat
            
            if filters:
                for key, value in filters.items():
                    query += f" AND {key} = %s"
                    params.append(value)
                    
            query += " LIMIT %s"
            params.append(limit)
            
            return self.db.execute_query(query, params)
            
        except Exception as e:
            logger.error(f"Doctor location search failed: {str(e)}")
            return []

    def _search_hospitals_nearby(self,
                               lat: float,
                               lon: float,
                               radius_km: float,
                               limit: int,
                               filters: Optional[Dict] = None) -> List[Dict]:
        """Search for hospitals near coordinates"""
        try:
            query = """
            SELECT *
            FROM hospitals
            WHERE ST_DWithin(
                ST_MakePoint(%s, %s)::geography,
                location::geography,
                %s * 1000
            )
            """
            params = [lon, lat, radius_km]
            
            if filters:
                for key, value in filters.items():
                    query += f" AND {key} = %s"
                    params.append(value)
                    
            query += " LIMIT %s"
            params.append(limit)
            
            return self.db.execute_query(query, params)
            
        except Exception as e:
            logger.error(f"Hospital location search failed: {str(e)}")
            return []

    def _search_clinics_nearby(self,
                             lat: float,
                             lon: float,
                             radius_km: float,
                             limit: int,
                             filters: Optional[Dict] = None) -> List[Dict]:
        """Search for clinics near coordinates"""
        try:
            query = """
            SELECT *
            FROM clinics
            WHERE ST_DWithin(
                ST_MakePoint(%s, %s)::geography,
                location::geography,
                %s * 1000
            )
            """
            params = [lon, lat, radius_km]
            
            if filters:
                for key, value in filters.items():
                    query += f" AND {key} = %s"
                    params.append(value)
                    
            query += " LIMIT %s"
            params.append(limit)
            
            return self.db.execute_query(query, params)
            
        except Exception as e:
            logger.error(f"Clinic location search failed: {str(e)}")
            return []

    def _log_location_search(self,
                           location: str,
                           entity_type: str,
                           radius_km: float,
                           results_count: int,
                           execution_time: float) -> None:
        """Log location search metrics"""
        try:
            self.db.execute("""
                INSERT INTO search_metrics
                (query, entity_type, total_results, execution_time, result_types)
                VALUES (%s, %s, %s, %s, %s)
            """, [
                f"location:{location}",
                entity_type,
                results_count,
                execution_time,
                json.dumps({
                    'radius_km': radius_km,
                    'location': location
                })
            ])
            
        except Exception as e:
            logger.error(f"Failed to log location search: {str(e)}")