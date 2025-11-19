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
        # Path goes up 3 levels: prediction.py -> services -> app -> backend -> project root
        self.models_dir = Path(__file__).parent.parent.parent.parent / "models"
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
            # Load XGBoost total pollen prediction model
            xgb_model_path = self.models_dir / "xgboost_total_pollen.joblib"
            if xgb_model_path.exists():
                self.models['main'] = joblib.load(xgb_model_path)
                logger.info(f"✅ Loaded XGBoost total pollen model: {xgb_model_path}")
            else:
                # Fallback to Random Forest model
                main_model_path = self.models_dir / "rf_enhanced_features_model.joblib"
                if main_model_path.exists():
                    self.models['main'] = joblib.load(main_model_path)
                    logger.info(f"✅ Loaded RF model (fallback): {main_model_path}")
            
            # Load allergen-specific XGBoost models
            allergen_models = {
                'tree': 'xgboost_tree.joblib',
                'grass': 'xgboost_grass.joblib',
                'weed': 'xgboost_weed.joblib',
                'ragweed': 'xgboost_ragweed.joblib'
            }
            
            for allergen_type, filename in allergen_models.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[allergen_type] = joblib.load(model_path)
                    logger.info(f"✅ Loaded {allergen_type} model: {filename}")
                else:
                    logger.warning(f"⚠️  {allergen_type} model not found: {filename}")
            
            logger.info(f"📊 Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def _engineer_features(self, weather_data: Dict, date: datetime, 
                          historical_pollen: Optional[List[float]] = None,
                          historical_temps: Optional[List[float]] = None,
                          historical_precip: Optional[List[float]] = None,
                          historical_wind: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Engineer features matching XGBoost total pollen model expectations
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
        
        # Weather features - Convert Fahrenheit to Celsius (model was trained on Celsius!)
        temp_max_f = weather_data.get('temp_max', 70.0)
        temp_min_f = weather_data.get('temp_min', 50.0)
        temp_avg_f = weather_data.get('temp_avg', (temp_max_f + temp_min_f) / 2)
        
        temp_max = (temp_max_f - 32) * 5/9
        temp_min = (temp_min_f - 32) * 5/9
        temp_avg = (temp_avg_f - 32) * 5/9
        
        precipitation = weather_data.get('precipitation', 0.0) * 25.4  # inches to mm
        wind_speed = weather_data.get('wind_speed', 5.0) * 1.609  # mph to km/h
        
        temp_range = temp_max - temp_min
        is_rainy = 1 if precipitation > 0 else 0
        
        # Temperature anomaly - use historical temps if provided
        if historical_temps and len(historical_temps) >= 7:
            # Convert historical temps from F to C
            temps_c = [(t - 32) * 5/9 for t in historical_temps[-30:]]
            tavg_30day = np.mean(temps_c)
            temp_anomaly = temp_avg - tavg_30day
        else:
            # Fallback to seasonal average
            seasonal_avg_f = self._get_seasonal_avg_temp(month)
            seasonal_avg = (seasonal_avg_f - 32) * 5/9
            temp_anomaly = temp_avg - seasonal_avg
        
        # Rainfall season cumsum - use historical precip if provided
        if historical_precip and len(historical_precip) > 0:
            # Convert inches to mm and sum
            rainfall_season_cumsum = sum(p * 25.4 for p in historical_precip) + precipitation
        else:
            # Simplified estimate
            season_start_doy = ((month - 1) // 3) * 91
            days_in_season = max(1, day_of_year - season_start_doy)
            rainfall_season_cumsum = precipitation * max(1, days_in_season / 7)
        
        # Wind percentile - use historical wind if provided
        if historical_wind and len(historical_wind) >= 7:
            # Convert mph to km/h
            winds_kmh = [w * 1.609 for w in historical_wind[-30:]]
            winds_kmh.append(wind_speed)
            # Calculate percentile
            wind_percentile = (sum(1 for w in winds_kmh if wind_speed >= w) / len(winds_kmh)) * 100
        else:
            # Simplified estimate
            wind_percentile = 50.0
            if wind_speed < 3 * 1.609:
                wind_percentile = 25.0
            elif wind_speed < 5 * 1.609:
                wind_percentile = 40.0
            elif wind_speed > 10 * 1.609:
                wind_percentile = 75.0
            elif wind_speed > 15 * 1.609:
                wind_percentile = 90.0
        
        # Lag features (if historical data provided)
        pollen_lag1 = historical_pollen[-1] if historical_pollen and len(historical_pollen) >= 1 else 50.0
        pollen_lag3 = historical_pollen[-3] if historical_pollen and len(historical_pollen) >= 3 else 50.0
        pollen_lag7 = historical_pollen[-7] if historical_pollen and len(historical_pollen) >= 7 else 50.0
        
        # Temperature and precipitation lags
        tmax_lag1 = temp_max  # Simplified
        tmax_lag3 = temp_max
        prcp_lag1 = precipitation
        prcp_lag3 = precipitation
        
        # Rolling features (simplified)
        pollen_roll3 = np.mean(historical_pollen[-3:]) if historical_pollen and len(historical_pollen) >= 3 else 50.0
        pollen_roll7 = np.mean(historical_pollen[-7:]) if historical_pollen and len(historical_pollen) >= 7 else 50.0
        
        temp_roll3 = temp_avg
        temp_roll7 = temp_avg
        rain_roll3 = precipitation * 3
        rain_roll7 = precipitation * 7
        
        # Growing degree days (species-specific)
        gdd_general = max(temp_avg - 5, 0)
        gdd_tree = max(temp_avg - 5, 0)  # 5°C base
        gdd_grass = max(temp_avg - 10, 0)  # 10°C base
        gdd_weed = max(temp_avg - 8, 0)  # 8°C base
        
        gdd_cumsum = gdd_general * day_of_year
        gdd_tree_cumsum = gdd_tree * day_of_year
        gdd_grass_cumsum = gdd_grass * day_of_year
        gdd_weed_cumsum = gdd_weed * day_of_year
        
        # Peak season proximity (Gaussian proximity to known pollen peaks)
        tree_peak_prox = np.exp(-0.5 * ((day_of_year - 110) / 30) ** 2)  # Mid-April
        grass_peak_prox = np.exp(-0.5 * ((day_of_year - 175) / 30) ** 2)  # Late June
        weed_peak_prox = np.exp(-0.5 * ((day_of_year - 255) / 30) ** 2)  # Mid-September
        
        # Interaction features
        temp_x_spring = temp_avg if month in [3, 4, 5] else 0
        temp_x_summer = temp_avg if month in [6, 7, 8] else 0
        rain_x_wind = precipitation * wind_speed
        
        # Create feature dictionary matching XGBoost model expectations
        features = {
            'TMAX': temp_max,
            'TMIN': temp_min,
            'TAVG': temp_avg,
            'PRCP': precipitation,
            'AWND': wind_speed,
            'Temp_Range': temp_range,
            'Is_Rainy': is_rainy,
            'Temp_Anomaly': temp_anomaly,
            'Rainfall_Season_Cumsum': rainfall_season_cumsum,
            'Wind_Percentile': wind_percentile,
            'Day_of_Week': day_of_week,
            'Month_sin': month_sin,
            'Month_cos': month_cos,
            'DOY_sin': doy_sin,
            'DOY_cos': doy_cos,
            'Year_Numeric': year,
            'Month_Numeric': month,
            'Pollen_lag_1': pollen_lag1,
            'TMAX_lag_1': tmax_lag1,
            'PRCP_lag_1': prcp_lag1,
            'Pollen_lag_3': pollen_lag3,
            'TMAX_lag_3': tmax_lag3,
            'PRCP_lag_3': prcp_lag3,
            'Pollen_lag_7': pollen_lag7,
            'Pollen_roll_3': pollen_roll3,
            'Temp_roll_3': temp_roll3,
            'Rain_roll_3': rain_roll3,
            'Pollen_roll_7': pollen_roll7,
            'Temp_roll_7': temp_roll7,
            'Rain_roll_7': rain_roll7,
            'GDD_General': gdd_general,
            'GDD_Tree': gdd_tree,
            'GDD_Grass': gdd_grass,
            'GDD_Weed': gdd_weed,
            'GDD_cumsum': gdd_cumsum,
            'GDD_Tree_cumsum': gdd_tree_cumsum,
            'GDD_Grass_cumsum': gdd_grass_cumsum,
            'GDD_Weed_cumsum': gdd_weed_cumsum,
            'Tree_Peak_Prox': tree_peak_prox,
            'Grass_Peak_Prox': grass_peak_prox,
            'Weed_Peak_Prox': weed_peak_prox,
            'Temp_x_Spring': temp_x_spring,
            'Temp_x_Summer': temp_x_summer,
            'Rain_x_Wind': rain_x_wind,
        }
        
        return pd.DataFrame([features])
    
    def _get_seasonal_avg_temp(self, month: int) -> float:
        """Get approximate seasonal average temperature"""
        seasonal_temps = {
            1: 32, 2: 35, 3: 45, 4: 57, 5: 67, 6: 76,
            7: 81, 8: 79, 9: 71, 10: 59, 11: 47, 12: 36
        }
        return seasonal_temps.get(month, 60)
    
    def _pollen_count_to_severity(self, pollen_count: float) -> float:
        """Convert raw pollen count to 0-10 severity scale based on percentile thresholds"""
        # Thresholds derived from historical pollen data percentiles
        # [0, 10, 25, 40, 55, 70, 80, 85, 90, 95, 100] percentiles
        thresholds = [0.6, 3.0, 7.0, 14.0, 29.0, 61.0, 131.0, 202.5, 341.3, 642.8]
        
        if pollen_count <= 0:
            return 0.0
        
        # Find which severity bracket the pollen count falls into
        for i in range(len(thresholds) - 1, -1, -1):
            if pollen_count >= thresholds[i]:
                return float(i)
        
        return 0.0
    
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
                     historical_pollen: Optional[List[float]] = None,
                     historical_temps: Optional[List[float]] = None,
                     historical_precip: Optional[List[float]] = None,
                     historical_wind: Optional[List[float]] = None) -> Tuple[float, str]:
        """
        Predict daily pollen severity
        
        Returns:
            Tuple of (severity_score, severity_level)
        """
        if 'main' not in self.models:
            self.load_models()
        
        # Engineer features
        features = self._engineer_features(
            weather_data, date, historical_pollen,
            historical_temps, historical_precip, historical_wind
        )
        
        # Log features for debugging
        logger.info(f"Date: {date.date()}, DOY: {date.timetuple().tm_yday}")
        logger.info(f"Pollen lags: lag1={features['Pollen_lag_1'].iloc[0]:.1f}, lag3={features['Pollen_lag_3'].iloc[0]:.1f}, lag7={features['Pollen_lag_7'].iloc[0]:.1f}")
        logger.info(f"Peak prox: Tree={features['Tree_Peak_Prox'].iloc[0]:.3f}, Grass={features['Grass_Peak_Prox'].iloc[0]:.3f}, Weed={features['Weed_Peak_Prox'].iloc[0]:.3f}")
        
        # Make prediction - XGBoost predicts severity (0-10) directly, not raw pollen count
        model = self.models['main']
        severity = float(model.predict(features)[0])
        
        logger.info(f"Raw severity prediction: {severity:.2f}")
        
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
    
    def identify_allergens(self, weather_data: Dict, date: datetime,
                          historical_pollen: Optional[List[float]] = None,
                          historical_temps: Optional[List[float]] = None,
                          historical_precip: Optional[List[float]] = None,
                          historical_wind: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Identify allergen contributions using XGBoost allergen-specific models
        
        Returns:
            Dictionary of allergen types to severity scores (0-10)
        """
        if 'main' not in self.models:
            self.load_models()
            
        # Check if we have allergen models loaded
        allergen_models_available = all(k in self.models for k in ['grass', 'tree', 'ragweed', 'weed'])
        
        if not allergen_models_available:
            # Fallback: estimate from main model using seasonal patterns
            logger.warning("Allergen models not available, using seasonal estimation")
            severity, _ = self.predict_daily(
                weather_data, date, historical_pollen, 
                historical_temps, historical_precip, historical_wind
            )
            
            # Estimate allergen distribution based on season
            month = date.month
            if month in [3, 4, 5]:  # Spring - Tree dominant
                return {'tree': severity * 0.6, 'grass': severity * 0.2, 
                       'ragweed': severity * 0.1, 'weed': severity * 0.1}
            elif month in [6, 7, 8]:  # Summer - Grass dominant
                return {'tree': severity * 0.1, 'grass': severity * 0.7,
                       'ragweed': severity * 0.1, 'weed': severity * 0.1}
            elif month in [9, 10]:  # Fall - Ragweed/Weed dominant
                return {'tree': severity * 0.1, 'grass': severity * 0.2,
                       'ragweed': severity * 0.5, 'weed': severity * 0.2}
            else:  # Winter - Low activity
                return {'tree': severity * 0.3, 'grass': severity * 0.1,
                       'ragweed': severity * 0.1, 'weed': severity * 0.5}
        
        # Use allergen-specific XGBoost models
        logger.info(f"Using allergen-specific models for {date.date()}")
        features = self._engineer_features(
            weather_data, date, historical_pollen,
            historical_temps, historical_precip, historical_wind
        )
        
        allergen_scores = {}
        for allergen in ['tree', 'grass', 'weed', 'ragweed']:
            if allergen in self.models:
                # Models predict severity (0-10) directly
                score = float(self.models[allergen].predict(features)[0])
                allergen_scores[allergen] = max(0.0, min(10.0, score))
                logger.info(f"{allergen.capitalize()} severity: {allergen_scores[allergen]:.2f}")
        
        return allergen_scores


# Global service instance
prediction_service = PredictionService()
