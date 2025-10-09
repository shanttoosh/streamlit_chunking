# Run FastAPI Application
import uvicorn
from src.api.main import app
from src.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
