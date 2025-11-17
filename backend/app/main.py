"""
FastAPI Main Application
Pollen severity prediction and allergen identification API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predictions, health
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pollen Predictor API",
    description="API for predicting daily pollen severity and identifying allergen drivers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Pollen Predictor API...")
    logger.info("📊 Loading ML models...")
    # Models will be loaded lazily in the prediction service

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down Pollen Predictor API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
