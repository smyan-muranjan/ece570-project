"""
Health check router
"""

from fastapi import APIRouter
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=True,
        available_endpoints=[
            "/health",
            "/api/v1/predict/daily",
            "/api/v1/predict/weekly",
            "/api/v1/allergen/identify"
        ]
    )
