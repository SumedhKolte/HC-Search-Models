from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os
import signal
import logging
from datetime import datetime
from pathlib import Path
from sqlalchemy import text
import asyncio
from src.tasks.update_embeddings import run_embedding_updates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

from src.core.engine import SearchEngine
from src.utils.metrics import PerformanceMetrics
from src.utils.validators import InputValidator
from src.data.database import Database
from config.settings import API_CONFIG

app = FastAPI(
    title="Medical Search API",
    description="Medical search service with semantic search capabilities",
    version="1.0.0"
)

# Initialize components
try:
    db = Database()
    metrics = PerformanceMetrics()
    validator = InputValidator()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store background tasks
app.state.background_tasks = set()

@app.on_event("startup")
async def startup_event():
    """Initialize application components"""
    try:
        # Initialize search engine
        app.state.search_engine = SearchEngine()
        await app.state.search_engine.initialize()
        
        logger.info("Search engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

@app.on_event("shutdown") 
async def shutdown_event():
    """Run cleanup on shutdown"""
    logger.info("Shutting down Medical Search API")
    
    # Cancel all background tasks
    for task in app.state.background_tasks:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Background task did not complete in time")
        except Exception as e:
            logger.error(f"Error cancelling task: {e}")
    
    # Cleanup database
    if hasattr(app.state, 'db'):
        app.state.db.dispose()

class SearchRequest(BaseModel):
    query: str
    search_type: str = "doctors"  # Changed from entity_type to search_type
    filters: Optional[Dict] = None
    location: Optional[str] = None
    radius_km: Optional[float] = None
    limit: int = 10

@app.post("/api/search")
async def search(request: SearchRequest):
    """Main search endpoint"""
    try:
        # Validate input
        is_valid, error = validator.validate_query(request.query)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
            
        # Execute search
        results = app.state.search_engine.search(
            query=request.query,
            search_type=request.search_type,  # Updated parameter name
            filters=request.filters,
            location=request.location,
            radius_km=request.radius_km,
            limit=request.limit
        )
        
        # Log metrics
        metrics.log_search_metrics(
            query=request.query,
            search_type=request.search_type,  # Updated parameter name
            results_count=len(results),
            execution_time=results.get('execution_time', 0),
            filters=request.filters
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/suggestions")
async def get_suggestions(query: str, entity_type: str):
    """Get search suggestions"""
    try:
        suggestions = app.state.search_engine.get_suggestions(
            query=query,
            entity_type=entity_type
        )
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection with simple query
        with db.engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.scalar()
        
        return {
            "status": "healthy",
            "database": "connected",
            "model": "loaded",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

if __name__ == "__main__":
    try:
        uvicorn.run(
            "src.api.server:app",
            host=API_CONFIG['host'],
            port=API_CONFIG['port'],
            reload=API_CONFIG.get('reload', False),
            workers=API_CONFIG.get('workers', 1),
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping server...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise
    finally:
        logger.info("Server shutdown complete")