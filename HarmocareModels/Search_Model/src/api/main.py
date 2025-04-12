from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from src.api.routes import search
from src.data.database import Database
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    try:
        # Startup
        logger.info("Starting up application...")
        db = Database()
        app.state.db = db
        yield
    finally:
        # Shutdown
        logger.info("Shutting down application...")
        try:
            await app.state.db.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

# Create FastAPI app
app = FastAPI(
    title="Medical Search API",
    description="API for searching medical entities",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with prefix
app.include_router(
    search.router,
    prefix="/api",
    tags=["search"]
)

# Health check endpoint
@app.get("/", tags=["health"])
async def health_check():
    """API health check endpoint"""
    return {"status": "healthy", "message": "Medical Search API is running"}

# Exception handler for SQLAlchemy errors
@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Database error occurred",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)