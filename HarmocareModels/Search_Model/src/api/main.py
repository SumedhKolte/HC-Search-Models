from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import search
from src.data.database import Database
from contextlib import asynccontextmanager
import logging

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

app = FastAPI(
    title="Medical Search API",
    description="API for searching medical entities using vector embeddings",
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

# Include routers
app.include_router(search.router, tags=["search"])

@app.get("/", tags=["health"])
async def health_check():
    """API health check endpoint"""
    return {"status": "healthy", "message": "Medical Search API is running"}