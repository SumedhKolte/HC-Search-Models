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
router = APIRouter()

# Initialize components
search_engine = SearchEngine()
metrics = PerformanceMetrics()

# Request models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    entity_types: List[str] = Field(
        default=['hospitals', 'doctors', 'clinics'],
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
        valid_types = {'doctors', 'hospitals', 'clinics'}
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

@router.post("/search/doctors")
async def search_doctors(request: SearchRequest, background_tasks: BackgroundTasks):
    """Search doctors endpoint"""
    try:
        if not request.query or request.query.isspace():
            return {
                "status": "error",
                "message": "Query cannot be empty",
                "results": []
            }
        
        search_result = await search_engine.search_entity(
            query=request.query,
            entity_type='doctors',
            filters=request.filters,
            limit=request.limit
        )
        
        return {
            "status": search_result['status'],
            "message": search_result.get('message'),
            "results": search_result['data']
        }
        
    except Exception as e:
        logger.error(f"Doctor search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/search/hospitals")
async def search_hospitals(request: SearchRequest, background_tasks: BackgroundTasks):
    """Search hospitals endpoint"""
    try:
        if not request.query or request.query.isspace():
            return {
                "status": "error",
                "message": "Query cannot be empty",
                "results": []
            }
        
        entity_results = await search_engine.search_entity(
            query=request.query,
            entity_type='hospitals',
            filters=request.filters,
            limit=request.limit
        )
        
        if not entity_results:
            return {
                "status": "success",
                "message": "No matching hospitals found for your query",
                "results": []
            }
            
        return {
            "status": "success",
            "results": entity_results
        }
        
    except Exception as e:
        logger.error(f"Hospital search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/search/clinics")
async def search_clinics(request: SearchRequest, background_tasks: BackgroundTasks):
    """Search clinics endpoint"""
    try:
        if not request.query or request.query.isspace():
            return {
                "status": "error", 
                "message": "Query cannot be empty",
                "results": []
            }
        
        entity_results = await search_engine.search_entity(
            query=request.query,
            entity_type='clinics',
            filters=request.filters,
            limit=request.limit
        )
        
        if not entity_results:
            return {
                "status": "success",
                "message": "No matching clinics found for your query",
                "results": []
            }
            
        return {
            "status": "success",
            "results": entity_results
        }
        
    except Exception as e:
        logger.error(f"Clinic search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/search")
async def unified_search(request: SearchRequest, background_tasks: BackgroundTasks):
    """Unified search across multiple entity types"""
    try:
        # Auto-detect entity type if not specified
        if not request.entity_types:
            inferred_type = search_engine.infer_entity_type(request.query)
            request.entity_types = [inferred_type]
            
        logger.info(f"Searching for entity types: {request.entity_types}")
        
        results = {}
        total_results = 0
        
        for entity_type in request.entity_types:
            try:
                # Normalize location filter
                filters = dict(request.filters)
                if filters.get('city'):
                    filters['location' if entity_type in ['hospitals', 'clinics'] else 'city'] = filters.pop('city')

                entity_results = await search_engine.search_entity(
                    query=request.query,
                    entity_type=entity_type,
                    filters=filters,
                    limit=request.limit
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
                "inferred_types": request.entity_types,
                "total_results": total_results,
                "query": request.query
            }
        }

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=str(e)
        )