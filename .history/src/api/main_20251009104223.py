# FastAPI Main Application
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import processing, database, retrieval, export, health, openai
# from ..config.settings import settings
# from ..config.logging import logger

# Create FastAPI app
app = FastAPI(
    title="Chunking Optimizer API", 
    version="2.0",
    description="Advanced text chunking and embedding API with multiple processing modes"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(processing.router, prefix="/api/v1", tags=["processing"])
app.include_router(database.router, prefix="/api/v1", tags=["database"])
app.include_router(retrieval.router, prefix="/api/v1", tags=["retrieval"])
app.include_router(export.router, prefix="/api/v1", tags=["export"])
app.include_router(openai.router, tags=["openai"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Chunking Optimizer API",
        "version": "2.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
