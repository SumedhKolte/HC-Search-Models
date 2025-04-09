import uvicorn
from config.settings import API_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the FastAPI server"""
    try:
        logger.info("Starting Medical Search API server...")
        uvicorn.run(
            "src.api.server:app",
            host=API_CONFIG['host'],
            port=API_CONFIG['port'],
            reload=API_CONFIG.get('reload', False),
            workers=API_CONFIG.get('workers', 1),
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise
    finally:
        logger.info("Server shutdown complete")

if __name__ == "__main__":
    main()