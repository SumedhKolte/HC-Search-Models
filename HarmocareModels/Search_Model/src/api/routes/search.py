import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(project_root))

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks, status
from typing import Dict, List, Optional, Annotated
from pydantic import BaseModel, Field
import logging

from src.core.search_engine import SearchEngine
from src.utils.metrics import PerformanceMetrics
from src.data.database import Database, get_db

logger = logging.getLogger(__name__)
router = APIRouter()
search_engine = SearchEngine()
metrics = PerformanceMetrics()

# Request models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    entity_types: List[str] = Field(default=['doctors', 'hospitals', 'clinics'])
    filters: Optional[Dict] = None
    limit: int = Field(default=10, ge=1, le=100)

@router.post("/doctors")
async def search_doctors(request: SearchRequest, background_tasks: BackgroundTasks):
    """Search doctors endpoint"""
    try:
        if not request.query or request.query.isspace():
            return {
                "status": "error",
                "message": "Query cannot be empty",
                "results": []
            }
        
        entity_results = await search_engine.search_entity(
            query=request.query,
            entity_type='doctors',
            filters=request.filters,
            limit=request.limit
        )
        
        if not entity_results:
            return {
                "status": "success",
                "message": "No matching doctors found for your query",
                "results": []
            }
            
        return {
            "status": "success",
            "results": entity_results
        }
        
    except Exception as e:
        logger.error(f"Doctor search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/hospitals")
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

@router.post("/clinics")
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

@router.get("/search")  # Add GET endpoint
@router.post("/search")  # Keep POST endpoint
async def unified_search(
    query: str,
    entity_types: List[str] = Query(default=['doctors']),
    limit: int = Query(default=10, ge=1, le=100),
    background_tasks: BackgroundTasks = None
):
    """Unified search across multiple entity types"""
    try:
        if not query or query.isspace():
            return {
                "status": "error",
                "message": "Query cannot be empty",
                "results": {}
            }
        
        results = {}
        
        for entity_type in entity_types:
            if entity_type not in ['doctors', 'hospitals', 'clinics', 'diseases', 'symptoms']:
                continue
                
            entity_results = await search_engine.search_entity(
                query=query,
                entity_type=entity_type,
                limit=limit
            )
            
            results[entity_type] = {
                "results": entity_results,
                "total": len(entity_results),
                "has_results": len(entity_results) > 0
            }
        
        if all(not v["has_results"] for v in results.values()):
            return {
                "status": "success", 
                "message": "No matching results found",
                "results": results
            }
            
        return {
            "status": "success",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )