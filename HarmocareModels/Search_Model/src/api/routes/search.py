import sys
from pathlib import Path
from datetime import datetime, timezone
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks, status
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import logging

from src.core.search_engine import SearchEngine
from src.utils.metrics import PerformanceMetrics
from src.data.database import Database, get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])  # Add tags for Swagger documentation

# Initialize components
search_engine = SearchEngine()
metrics = PerformanceMetrics()

# Request models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    entity_types: List[str] = Field(
        default=['hospitals', 'doctors', 'clinics', 'symptoms', 'diseases'],
        description="Types of entities to search"
    )
    filters: Optional[Dict] = Field(
        default={},
        description="Search filters including city"
    )
    limit: int = Field(default=10, ge=1, le=100)

    @field_validator('entity_types')
    @classmethod
    def validate_entity_types(cls, v: List[str]) -> List[str]:
        valid_types = {'doctors', 'hospitals', 'clinics', 'symptoms', 'diseases'}
        if not all(t in valid_types for t in v):
            raise ValueError(f"Invalid entity type. Must be one of: {valid_types}")
        return v

    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v: Optional[Dict]) -> Dict:
        if v is None:
            return {}
        
        # Normalize city name if provided
        if 'city' in v:
            if not v['city']:
                raise ValueError("City filter cannot be empty if provided")
            # Normalize city name
            v['city'] = v['city'].strip().lower()
            
            # Optional: Add validation against known cities
            valid_cities = {'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata'}
            if v['city'] not in valid_cities:
                raise ValueError(f"Invalid city. Must be one of: {valid_cities}")
        
        return v

# Add Query parameters model for GET requests
class SearchParams:
    def __init__(
        self,
        query: str = Query(..., min_length=1, max_length=500),
        city: Optional[str] = Query(None, description="City filter"),
        limit: int = Query(10, ge=1, le=100),
        hospital_type: Optional[str] = Query(None),
        specialization: Optional[str] = Query(None)
    ):
        self.query = query
        self.filters = {}
        if city:
            self.filters['city'] = city
        if hospital_type:
            self.filters['hospital_type'] = hospital_type
        if specialization:
            self.filters['specialization'] = specialization
        self.limit = limit

async def perform_search(
    query: str,
    entity_types: List[str],
    filters: Dict,
    limit: int,
    background_tasks: BackgroundTasks
) -> Dict:
    """Common search logic for both GET and POST requests"""
    try:
        results = {}
        total_results = 0
        
        for entity_type in entity_types:
            try:
                current_filters = dict(filters)
                
                # Only apply location filters to physical entities
                if entity_type in ['symptoms', 'diseases']:
                    current_filters = {}  # No filters for symptoms/diseases
                elif current_filters.get('city'):
                    current_filters['location' if entity_type in ['hospitals', 'clinics'] else 'city'] = current_filters.pop('city')

                entity_results = await search_engine.search_entity(
                    query=query,
                    entity_type=entity_type,
                    filters=current_filters,
                    limit=limit
                )
                
                results[entity_type] = {
                    "results": entity_results,
                    "total": len(entity_results),
                    "has_results": len(entity_results) > 0
                }
                
                total_results += len(entity_results)
                
            except Exception as e:
                logger.error(f"Error searching {entity_type}: {str(e)}")
                results[entity_type] = {
                    "results": [],
                    "total": 0,
                    "error": str(e)
                }

        return {
            "status": "success",
            "results": results,
            "metadata": {
                "total_results": total_results,
                "query": query,
                "entity_types": entity_types
            }
        }
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 1. Unified Search Routes
@router.get("/search/unified", response_model=Dict)
@router.post("/search/unified", response_model=Dict)
async def unified_search(
    request: Optional[SearchRequest] = None,
    query: str = Query(None, min_length=1, max_length=500),
    city: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    background_tasks: BackgroundTasks = None
):
    """Unified search across all entity types"""
    if request:  # POST request
        search_params = request
    else:  # GET request
        search_params = SearchParams(query=query, city=city, limit=limit)
        
    return await perform_search(
        query=search_params.query,
        entity_types=['hospitals', 'doctors', 'clinics', 'symptoms', 'diseases'],
        filters=search_params.filters,
        limit=search_params.limit,
        background_tasks=background_tasks
    )

# 2. Doctors Search Routes
@router.get("/search/doctors", response_model=Dict)
@router.post("/search/doctors", response_model=Dict)
async def search_doctors(
    request: Optional[SearchRequest] = None,
    query: str = Query(None, min_length=1, max_length=500),
    city: Optional[str] = Query(None),
    specialization: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    background_tasks: BackgroundTasks = None
):
    """Search doctors and related medical conditions"""
    if request:  # POST request
        search_params = request
    else:  # GET request
        search_params = SearchParams(
            query=query,
            city=city,
            specialization=specialization,
            limit=limit
        )
    
    return await perform_search(
        query=search_params.query,
        entity_types=['doctors', 'symptoms', 'diseases'],
        filters=search_params.filters,
        limit=search_params.limit,
        background_tasks=background_tasks
    )

# 3. Medical Facilities Search Routes
@router.get("/search/medical-facilities", response_model=Dict)
@router.post("/search/medical-facilities", response_model=Dict)
async def search_facilities(
    request: Optional[SearchRequest] = None,
    query: str = Query(None, min_length=1, max_length=500),
    city: Optional[str] = Query(None),
    hospital_type: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    background_tasks: BackgroundTasks = None
):
    """Search hospitals, clinics and related medical conditions"""
    if request:  # POST request
        search_params = request
    else:  # GET request
        search_params = SearchParams(
            query=query,
            city=city,
            hospital_type=hospital_type,
            limit=limit
        )
    
    return await perform_search(
        query=search_params.query,
        entity_types=['hospitals', 'clinics', 'symptoms', 'diseases'],
        filters=search_params.filters,
        limit=search_params.limit,
        background_tasks=background_tasks
    )