from pathlib import Path
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.exc import SQLAlchemyError 
from sqlalchemy.orm import sessionmaker
import logging
from datetime import datetime, timezone
import ssl

from .db_config import get_db_config, get_async_db_url

logger = logging.getLogger(__name__)

class Database:
    """Database connection and query handler"""
    def __init__(self):
        """Initialize database connection"""
        self.db_config = get_db_config()
        self.engine: Optional[AsyncEngine] = None
        self.SessionLocal = None
        self._setup_connection()

    def _build_connection_string(self) -> tuple[str, dict]:
        """Build PostgreSQL connection string with SSL config"""
        base_url = get_async_db_url()
        
        # Configure SSL context for asyncpg
        ssl_context = ssl.create_default_context(
            cafile=self.db_config['sslcert']
        )
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        connect_args = {
            'ssl': ssl_context,
            'server_settings': {'application_name': 'medical_search'}
        }
        
        return base_url, connect_args

    def _setup_connection(self):
        """Setup async database connection with SSL"""
        try:
            base_url, connect_args = self._build_connection_string()
            
            self.engine = create_async_engine(
                base_url,
                connect_args=connect_args,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                echo=False
            )
            
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {str(e)}")
            raise

    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute async database query"""
        try:
            async with self.SessionLocal() as session:
                async with session.begin():
                    if isinstance(query, str):
                        query = text(query)
                    
                    # Ensure params is a dictionary
                    params = params or {}
                    
                    result = await session.execute(query, params)
                    return [dict(row) for row in result.mappings()]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def get_session(self) -> AsyncSession:
        """Get database session as async context manager"""
        return self.SessionLocal()

    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        async with self.get_session() as session:
            try:
                yield session
            finally:
                await session.close()

    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()

# Global database instance
_db = None

def get_db() -> Database:
    """Get global database instance"""
    global _db
    if _db is None:
        _db = Database()
    return _db

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    db = get_db()
    async with db.get_session() as session:
        try:
            yield session
        finally:
            await session.close()
