"""
Pydantic schemas for API request and response models
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import date as date_type, datetime
from enum import Enum


class SeverityLevel(str, Enum):
    """Pollen severity level categories"""
    NONE = "None"
    VERY_LOW = "Very Low"
    LOW = "Low"
    LOW_MODERATE = "Low-Moderate"
    MODERATE = "Moderate"
    MODERATE_HIGH = "Moderate-High"
    HIGH = "High"
    VERY_HIGH = "Very High"
    EXTREME = "Extreme"
    SEVERE = "Severe"


class AllergenType(str, Enum):
    """Types of allergens"""
    TREE = "Tree"
    GRASS = "Grass"
    RAGWEED = "Ragweed"
    WEED = "Weed"


class WeatherInput(BaseModel):
    """Weather data input for prediction"""
    date: date_type = Field(..., description="Date for prediction")
    temp_max: float = Field(..., ge=-50, le=150, description="Maximum temperature (°F)")
    temp_min: float = Field(..., ge=-50, le=150, description="Minimum temperature (°F)")
    temp_avg: Optional[float] = Field(None, ge=-50, le=150, description="Average temperature (°F)")
    precipitation: float = Field(0.0, ge=0, description="Precipitation (inches)")
    wind_speed: Optional[float] = Field(None, ge=0, description="Wind speed (mph)")
    
    @validator('temp_avg', pre=True, always=True)
    def calculate_temp_avg(cls, v, values):
        """Calculate average temperature if not provided"""
        if v is None and 'temp_max' in values and 'temp_min' in values:
            return (values['temp_max'] + values['temp_min']) / 2
        return v


class DailyPredictionRequest(BaseModel):
    """Request for daily pollen prediction"""
    weather: WeatherInput
    historical_pollen: Optional[List[float]] = Field(
        None, 
        description="Historical pollen counts for lag features (last 7 days, in raw counts)"
    )
    historical_temps: Optional[List[float]] = Field(
        None,
        description="Historical average temperatures for last 30 days (°F)"
    )
    historical_precip: Optional[List[float]] = Field(
        None,
        description="Historical precipitation for season-to-date (inches)"
    )
    historical_wind: Optional[List[float]] = Field(
        None,
        description="Historical wind speeds for last 30 days (mph)"
    )


class WeeklyPredictionRequest(BaseModel):
    """Request for weekly pollen forecast"""
    weather_forecast: List[WeatherInput] = Field(
        ..., 
        min_items=1, 
        max_items=7,
        description="Weather forecast for upcoming days (1-7 days)"
    )
    current_pollen: Optional[float] = Field(
        None,
        description="Current pollen count for lag features"
    )


class PollenPrediction(BaseModel):
    """Single pollen prediction result"""
    date: date_type
    severity_score: float = Field(..., ge=0, le=10, description="Pollen severity (0-10 scale)")
    severity_level: SeverityLevel
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence")
    
    class Config:
        use_enum_values = True


class DailyPredictionResponse(BaseModel):
    """Response for daily prediction"""
    prediction: PollenPrediction
    recommendation: str = Field(..., description="Health recommendation based on severity")
    

class WeeklyPredictionResponse(BaseModel):
    """Response for weekly forecast"""
    predictions: List[PollenPrediction]
    peak_day: Optional[date_type] = Field(None, description="Day with highest predicted pollen")
    summary: str = Field(..., description="Weekly forecast summary")


class AllergenPrediction(BaseModel):
    """Allergen-specific prediction"""
    allergen_type: AllergenType
    severity_score: float = Field(..., ge=0, le=10)
    severity_level: SeverityLevel
    contribution_pct: float = Field(..., ge=0, le=100, description="Percentage contribution to total pollen")
    
    class Config:
        use_enum_values = True


class AllergenIdentificationRequest(BaseModel):
    """Request for identifying primary allergen drivers"""
    weather: WeatherInput
    historical_pollen: Optional[List[float]] = Field(
        None, 
        description="Historical pollen counts for lag features (last 7 days, in raw counts)"
    )
    historical_temps: Optional[List[float]] = Field(
        None,
        description="Historical average temperatures for last 30 days (°F)"
    )
    historical_precip: Optional[List[float]] = Field(
        None,
        description="Historical precipitation for season-to-date (inches)"
    )
    historical_wind: Optional[List[float]] = Field(
        None,
        description="Historical wind speeds for last 30 days (mph)"
    )
    

class AllergenIdentificationResponse(BaseModel):
    """Response with allergen breakdown"""
    date: date_type
    total_severity: float = Field(..., ge=0, le=10)
    allergens: List[AllergenPrediction]
    primary_allergen: AllergenType
    alert_level: str = Field(..., description="Alert level (Low/Moderate/High/Severe)")
    
    class Config:
        use_enum_values = True


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    available_endpoints: List[str]
