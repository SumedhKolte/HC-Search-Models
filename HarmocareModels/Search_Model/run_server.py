import uvicorn
from config.settings import API_CONFIG

if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=API_CONFIG['reload'],
        workers=API_CONFIG['workers']
    )