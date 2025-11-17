"""
Prediction endpoints router
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import List
import logging

from app.schemas import (
    DailyPredictionRequest,
    DailyPredictionResponse,
    WeeklyPredictionRequest,
    WeeklyPredictionResponse,
    AllergenIdentificationRequest,
    AllergenIdentificationResponse,
    PollenPrediction,
    AllergenPrediction,
    SeverityLevel,
    AllergenType
)
from app.services.prediction import prediction_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict/daily", response_model=DailyPredictionResponse)
async def predict_daily_pollen(request: DailyPredictionRequest):
    """
    Predict daily pollen severity based on weather conditions
    
    - **weather**: Weather data for the prediction day
    - **historical_pollen**: Optional historical pollen counts for lag features
    """
    try:
        # Convert weather input to dict
        weather_dict = {
            'temp_max': request.weather.temp_max,
            'temp_min': request.weather.temp_min,
            'temp_avg': request.weather.temp_avg,
            'precipitation': request.weather.precipitation,
            'wind_speed': request.weather.wind_speed or 5.0
        }
        
        # Make prediction
        severity, level = prediction_service.predict_daily(
            weather_data=weather_dict,
            date=datetime.combine(request.weather.date, datetime.min.time()),
            historical_pollen=request.historical_pollen
        )
        
        # Get recommendation
        recommendation = prediction_service._get_recommendation(severity)
        
        prediction = PollenPrediction(
            date=request.weather.date,
            severity_score=round(severity, 2),
            severity_level=level,
            confidence=0.86  # Based on model R² score
        )
        
        return DailyPredictionResponse(
            prediction=prediction,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Error in daily prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/weekly", response_model=WeeklyPredictionResponse)
async def predict_weekly_pollen(request: WeeklyPredictionRequest):
    """
    Predict weekly pollen forecast
    
    - **weather_forecast**: List of weather forecasts for upcoming days (1-7 days)
    - **current_pollen**: Optional current pollen count for lag features
    """
    try:
        # Convert weather forecasts to list of dicts
        weather_list = []
        for weather in request.weather_forecast:
            weather_list.append({
                'date': datetime.combine(weather.date, datetime.min.time()),
                'temp_max': weather.temp_max,
                'temp_min': weather.temp_min,
                'temp_avg': weather.temp_avg,
                'precipitation': weather.precipitation,
                'wind_speed': weather.wind_speed or 5.0
            })
        
        # Make weekly predictions
        predictions_raw = prediction_service.predict_weekly(
            weather_forecast=weather_list,
            current_pollen=request.current_pollen
        )
        
        # Convert to response format
        predictions = []
        max_severity = 0
        peak_day = None
        
        for date, severity, level in predictions_raw:
            predictions.append(PollenPrediction(
                date=date.date(),
                severity_score=round(severity, 2),
                severity_level=level,
                confidence=0.86
            ))
            
            if severity > max_severity:
                max_severity = severity
                peak_day = date.date()
        
        # Generate summary
        avg_severity = sum(p.severity_score for p in predictions) / len(predictions)
        if avg_severity <= 3:
            summary = "Low pollen levels expected throughout the week."
        elif avg_severity <= 5:
            summary = "Moderate pollen levels expected. Monitor daily forecasts."
        elif avg_severity <= 7:
            summary = "High pollen levels expected. Plan indoor activities."
        else:
            summary = "Very high pollen levels expected. Take precautions."
        
        return WeeklyPredictionResponse(
            predictions=predictions,
            peak_day=peak_day,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in weekly prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Weekly prediction failed: {str(e)}")


@router.post("/allergen/identify", response_model=AllergenIdentificationResponse)
async def identify_allergens(request: AllergenIdentificationRequest):
    """
    Identify primary allergen drivers for the day
    
    - **weather**: Weather data for allergen identification
    """
    try:
        # Convert weather input to dict
        weather_dict = {
            'temp_max': request.weather.temp_max,
            'temp_min': request.weather.temp_min,
            'temp_avg': request.weather.temp_avg,
            'precipitation': request.weather.precipitation,
            'wind_speed': request.weather.wind_speed or 5.0
        }
        
        # Identify allergens
        allergen_scores = prediction_service.identify_allergens(
            weather_data=weather_dict,
            date=datetime.combine(request.weather.date, datetime.min.time())
        )
        
        # Calculate total and percentages
        total_severity = sum(allergen_scores.values())
        
        # Build allergen predictions
        allergens = []
        primary_allergen = None
        max_score = 0
        
        allergen_type_map = {
            'tree': AllergenType.TREE,
            'grass': AllergenType.GRASS,
            'ragweed': AllergenType.RAGWEED,
            'weed': AllergenType.WEED
        }
        
        for allergen_name, score in allergen_scores.items():
            if score > max_score:
                max_score = score
                primary_allergen = allergen_type_map[allergen_name]
            
            contribution_pct = (score / total_severity * 100) if total_severity > 0 else 0
            
            allergens.append(AllergenPrediction(
                allergen_type=allergen_type_map[allergen_name],
                severity_score=round(score, 2),
                severity_level=prediction_service._severity_to_level(score),
                contribution_pct=round(contribution_pct, 1)
            ))
        
        # Sort by severity
        allergens.sort(key=lambda x: x.severity_score, reverse=True)
        
        # Get alert level
        alert_level = prediction_service._get_alert_level(total_severity)
        
        return AllergenIdentificationResponse(
            date=request.weather.date,
            total_severity=round(total_severity, 2),
            allergens=allergens,
            primary_allergen=primary_allergen or AllergenType.GRASS,
            alert_level=alert_level
        )
        
    except Exception as e:
        logger.error(f"Error in allergen identification: {e}")
        raise HTTPException(status_code=500, detail=f"Allergen identification failed: {str(e)}")
