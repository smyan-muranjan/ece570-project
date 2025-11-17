"""
Prediction service for loading models and generating predictions
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for pollen prediction using trained ML models"""
    
    def __init__(self, models_dir: str = "../models"):
        """Initialize prediction service with model directory"""
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.models: Dict = {}
        self.feature_names: Optional[List[str]] = None
        self.pollen_thresholds = self._get_pollen_thresholds()
        
    def _get_pollen_thresholds(self) -> List[float]:
        """Return pollen severity thresholds (0-10 scale)"""
        # Based on percentiles from training data
        return [0.0, 10.0, 25.0, 40.0, 55.0, 70.0, 80.0, 85.0, 90.0, 95.0, 100.0]
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load main pollen prediction model
            main_model_path = self.models_dir / "rf_enhanced_features_model.joblib"
            if main_model_path.exists():
                self.models['main'] = joblib.load(main_model_path)
                logger.info(f"✅ Loaded main model: {main_model_path}")
            else:
                # Fallback to alternative model
                alt_model_path = self.models_dir / "rf_enhanced_features.joblib"
                if alt_model_path.exists():
                    self.models['main'] = joblib.load(alt_model_path)
                    logger.info(f"✅ Loaded main model (fallback): {alt_model_path}")
            
            # Load allergen-specific models
            allergen_models = {
                'grass': 'xgboost_grass.joblib',
                'tree': 'xgboost_tree.joblib',
                'ragweed': 'xgboost_ragweed.joblib',
                'weed': 'xgboost_weed.joblib'
            }
            
            for allergen, filename in allergen_models.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[allergen] = joblib.load(model_path)
                    logger.info(f"✅ Loaded {allergen} model")
            
            logger.info(f"📊 Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def _engineer_features(self, weather_data: Dict, date: datetime, 
                          historical_pollen: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Engineer features from weather data matching training features
        """
        # Extract date components
        year = date.year
        month = date.month
        day_of_year = date.timetuple().tm_yday
        day_of_week = date.weekday()
        
        # Cyclical encoding
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        doy_sin = np.sin(2 * np.pi * day_of_year / 365)
        doy_cos = np.cos(2 * np.pi * day_of_year / 365)
        
        # Weather features
        temp_max = weather_data.get('temp_max', 70.0)
        temp_min = weather_data.get('temp_min', 50.0)
        temp_avg = weather_data.get('temp_avg', (temp_max + temp_min) / 2)
        precipitation = weather_data.get('precipitation', 0.0)
        wind_speed = weather_data.get('wind_speed', 5.0)
        
        temp_range = temp_max - temp_min
        is_rainy = 1 if precipitation > 0 else 0
        
        # Temperature anomaly (simplified - using seasonal average)
        seasonal_avg = self._get_seasonal_avg_temp(month)
        temp_anomaly = temp_avg - seasonal_avg
        
        # Lag features (if historical data provided)
        pollen_lag1 = historical_pollen[-1] if historical_pollen and len(historical_pollen) >= 1 else 50.0
        pollen_lag3 = historical_pollen[-3] if historical_pollen and len(historical_pollen) >= 3 else 50.0
        pollen_lag7 = historical_pollen[-7] if historical_pollen and len(historical_pollen) >= 7 else 50.0
        
        # Rolling features (simplified)
        pollen_roll_mean3 = np.mean(historical_pollen[-3:]) if historical_pollen and len(historical_pollen) >= 3 else 50.0
        pollen_roll_mean7 = np.mean(historical_pollen[-7:]) if historical_pollen and len(historical_pollen) >= 7 else 50.0
        
        # Create feature dictionary
        features = {
            'Year': year,
            'Day_of_Year': day_of_year,
            'Day_of_Week': day_of_week,
            'Month_sin': month_sin,
            'Month_cos': month_cos,
            'DOY_sin': doy_sin,
            'DOY_cos': doy_cos,
            'TMAX': temp_max,
            'TMIN': temp_min,
            'TAVG': temp_avg,
            'PRCP': precipitation,
            'AWND': wind_speed,
            'Temp_Range': temp_range,
            'Is_Rainy': is_rainy,
            'Temp_Anomaly': temp_anomaly,
            'Total_Pollen_lag1': pollen_lag1,
            'Total_Pollen_lag3': pollen_lag3,
            'Total_Pollen_lag7': pollen_lag7,
            'Total_Pollen_roll_mean3': pollen_roll_mean3,
            'Total_Pollen_roll_mean7': pollen_roll_mean7,
        }
        
        return pd.DataFrame([features])
    
    def _get_seasonal_avg_temp(self, month: int) -> float:
        """Get approximate seasonal average temperature"""
        seasonal_temps = {
            1: 32, 2: 35, 3: 45, 4: 57, 5: 67, 6: 76,
            7: 81, 8: 79, 9: 71, 10: 59, 11: 47, 12: 36
        }
        return seasonal_temps.get(month, 60)
    
    def _severity_to_level(self, severity: float) -> str:
        """Convert severity score to level name"""
        levels = [
            "None", "Very Low", "Low", "Low-Moderate", "Moderate",
            "Moderate-High", "High", "Very High", "Extreme", "Severe"
        ]
        index = min(int(severity), 9)
        return levels[index]
    
    def _get_recommendation(self, severity: float) -> str:
        """Get health recommendation based on severity"""
        if severity <= 2:
            return "Low pollen levels. Enjoy outdoor activities!"
        elif severity <= 4:
            return "Moderate pollen. Sensitive individuals should monitor symptoms."
        elif severity <= 6:
            return "High pollen levels. Consider limiting outdoor exposure."
        elif severity <= 8:
            return "Very high pollen. Allergy sufferers should stay indoors when possible."
        else:
            return "Extreme pollen levels. Avoid outdoor activities and keep windows closed."
    
    def _get_alert_level(self, severity: float) -> str:
        """Get alert level based on severity"""
        if severity <= 3:
            return "Low"
        elif severity <= 5:
            return "Moderate"
        elif severity <= 7:
            return "High"
        else:
            return "Severe"
    
    def predict_daily(self, weather_data: Dict, date: datetime,
                     historical_pollen: Optional[List[float]] = None) -> Tuple[float, str]:
        """
        Predict daily pollen severity
        
        Returns:
            Tuple of (severity_score, severity_level)
        """
        if 'main' not in self.models:
            self.load_models()
        
        # Engineer features
        features = self._engineer_features(weather_data, date, historical_pollen)
        
        # Make prediction
        model = self.models['main']
        severity = float(model.predict(features)[0])
        
        # Ensure within 0-10 range
        severity = max(0.0, min(10.0, severity))
        level = self._severity_to_level(severity)
        
        return severity, level
    
    def predict_weekly(self, weather_forecast: List[Dict], 
                      current_pollen: Optional[float] = None) -> List[Tuple[datetime, float, str]]:
        """
        Predict weekly pollen forecast
        
        Returns:
            List of (date, severity, level) tuples
        """
        predictions = []
        historical = [current_pollen] if current_pollen else []
        
        for day_weather in weather_forecast:
            date = day_weather['date']
            weather = {k: v for k, v in day_weather.items() if k != 'date'}
            
            severity, level = self.predict_daily(weather, date, historical)
            predictions.append((date, severity, level))
            
            # Update historical with prediction for next day
            # Convert severity back to approximate pollen count
            pollen_estimate = severity * 10  # Simplified conversion
            historical.append(pollen_estimate)
            if len(historical) > 7:
                historical.pop(0)
        
        return predictions
    
    def identify_allergens(self, weather_data: Dict, date: datetime) -> Dict[str, float]:
        """
        Identify allergen contributions
        
        Returns:
            Dictionary of allergen types to severity scores
        """
        if not all(k in self.models for k in ['grass', 'tree', 'ragweed', 'weed']):
            # If allergen models not available, estimate from main model
            severity, _ = self.predict_daily(weather_data, date)
            
            # Estimate allergen distribution based on season
            month = date.month
            if month in [3, 4, 5]:  # Spring
                return {'tree': severity * 0.6, 'grass': severity * 0.2, 
                       'ragweed': severity * 0.1, 'weed': severity * 0.1}
            elif month in [6, 7, 8]:  # Summer
                return {'tree': severity * 0.1, 'grass': severity * 0.7,
                       'ragweed': severity * 0.1, 'weed': severity * 0.1}
            elif month in [9, 10]:  # Fall
                return {'tree': severity * 0.1, 'grass': severity * 0.2,
                       'ragweed': severity * 0.5, 'weed': severity * 0.2}
            else:  # Winter
                return {'tree': severity * 0.3, 'grass': severity * 0.1,
                       'ragweed': severity * 0.1, 'weed': severity * 0.5}
        
        # Use allergen-specific models
        features = self._engineer_features(weather_data, date)
        allergen_scores = {}
        
        for allergen in ['grass', 'tree', 'ragweed', 'weed']:
            if allergen in self.models:
                score = float(self.models[allergen].predict(features)[0])
                allergen_scores[allergen] = max(0.0, min(10.0, score))
        
        return allergen_scores


# Global service instance
prediction_service = PredictionService()
