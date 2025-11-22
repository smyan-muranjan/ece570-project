"""
Prediction service for loading models and generating predictions
Updated to use weather-only trained models for optimal performance
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
        """Load weather-only trained models (proven 47.9% better performance)"""
        try:
            # Load weather-only trained XGBoost models (bio_v2 series)
            # These models are trained exclusively on weather features and perform
            # significantly better for weather-only predictions
            
            # Load total pollen model
            weather_only_total_path = self.models_dir / "xgboost_total_pollen_bio_v2.joblib"
            if weather_only_total_path.exists():
                self.models['main'] = joblib.load(weather_only_total_path)
                logger.info(f"✅ Loaded weather-only total pollen model: {weather_only_total_path}")
            else:
                # Fallback to multitype model if weather-only not available
                fallback_path = self.models_dir / "xgboost_total_pollen.joblib"
                if fallback_path.exists():
                    self.models['main'] = joblib.load(fallback_path)
                    logger.warning(f"⚠️  Using multitype model (fallback): {fallback_path}")
                else:
                    # Final fallback to RF model
                    rf_path = self.models_dir / "rf_enhanced_features_model.joblib"
                    if rf_path.exists():
                        self.models['main'] = joblib.load(rf_path)
                        logger.warning(f"⚠️  Using RF model (final fallback): {rf_path}")
            
            # Load weather-only trained allergen-specific models
            allergen_models = {
                'tree': 'xgboost_tree_bio_v2.joblib',
                'grass': 'xgboost_grass_bio_v2.joblib',
                'weed': 'xgboost_weed_bio_v2.joblib',
                'ragweed': 'xgboost_ragweed_bio_v2.joblib'
            }
            
            for allergen_type, filename in allergen_models.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[allergen_type] = joblib.load(model_path)
                    logger.info(f"✅ Loaded weather-only {allergen_type} model: {filename}")
                else:
                    # Fallback to multitype model
                    fallback_filename = f"xgboost_{allergen_type}.joblib"
                    fallback_path = self.models_dir / fallback_filename
                    if fallback_path.exists():
                        self.models[allergen_type] = joblib.load(fallback_path)
                        logger.warning(f"⚠️  Using multitype {allergen_type} model (fallback): {fallback_filename}")
                    else:
                        logger.warning(f"⚠️  {allergen_type} model not found: {filename}")
            
            # Load feature list for weather-only models
            feature_list_path = self.models_dir / "model_features_list.joblib"
            if feature_list_path.exists():
                self.feature_names = joblib.load(feature_list_path)
                logger.info(f"✅ Loaded feature list: {len(self.feature_names)} features")
            else:
                logger.warning("⚠️  Feature list not found, using default feature engineering")
            
            logger.info(f"📊 Loaded {len(self.models)} weather-optimized models successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def _engineer_features(self, weather_data: Dict, date: datetime, 
                          historical_pollen: Optional[List[float]] = None,
                          historical_temps: Optional[List[float]] = None,
                          historical_precip: Optional[List[float]] = None,
                          historical_wind: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Engineer features matching weather-only trained model expectations
        Uses advanced biological features (VPD, Ventilation, Shock Index, etc.)
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
        
        # Weather features - Convert Fahrenheit to Celsius
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
        
        # --- ADVANCED BIOLOGICAL FEATURES ---
        
        # 1. Vapor Pressure Deficit (VPD) - The Release Trigger
        # Critical for anther dehiscence (pollen release)
        es = 0.6108 * np.exp((17.27 * temp_avg) / (temp_avg + 237.3))  # Saturation vapor pressure
        ea = 0.6108 * np.exp((17.27 * temp_min) / (temp_min + 237.3))  # Actual vapor pressure (using T_min as dew point proxy)
        vpd = max(0, es - ea)
        
        # 2. Ventilation Index - The Dispersion Engine
        # Combined effect of vertical mixing (diurnal range) and horizontal transport (wind)
        ventilation_index = wind_speed * temp_range
        
        # 3. Precipitation Dynamics
        # Days Since Rain (critical for post-rain release spikes)
        # For API, we'll estimate based on current precipitation
        if precipitation > 0:
            days_since_rain = 0
        else:
            # Estimate based on season (simplified)
            if month in [6, 7, 8]:  # Summer - typically drier
                days_since_rain = 3
            elif month in [12, 1, 2]:  # Winter - more frequent precipitation
                days_since_rain = 1
            else:
                days_since_rain = 2
        
        # Rain Intensity Categories
        if precipitation == 0:
            rain_light = 0
            rain_heavy = 0
        elif precipitation <= 2.5:  # Light rain
            rain_light = 1
            rain_heavy = 0
        else:  # Heavy rain
            rain_light = 0
            rain_heavy = 1
        
        # 4. Osmotic Shock Index (Thunderstorm Asthma Risk)
        # High VPD (dried pollen) + Light Rain (moisture) + Wind (fragmentation)
        shock_index = rain_light * wind_speed * vpd
        
        # 5. Growing Degree Days & Burst
        gdd_general = max(temp_avg - 5, 0)
        gdd_tree = max(temp_avg - 5, 0)
        gdd_grass = max(temp_avg - 10, 0)
        gdd_weed = max(temp_avg - 8, 0)
        
        # "Burst" - Rapid warming over last 5 days (simplified for API)
        gdd_burst_5d = gdd_general * 5  # Simplified estimate
        
        # Cumulative GDD (season progress)
        gdd_cumsum = gdd_general * day_of_year
        
        # --- WEATHER LAGS (Weather history is permitted) ---
        # Use historical data if provided, otherwise use current values as estimates
        
        # Temperature lags
        tmax_lag_1 = temp_max if not historical_temps or len(historical_temps) < 1 else (historical_temps[-1] - 32) * 5/9
        tmax_lag_2 = temp_max if not historical_temps or len(historical_temps) < 2 else (historical_temps[-2] - 32) * 5/9
        tmax_lag_3 = temp_max if not historical_temps or len(historical_temps) < 3 else (historical_temps[-3] - 32) * 5/9
        tmax_lag_7 = temp_max if not historical_temps or len(historical_temps) < 7 else (historical_temps[-7] - 32) * 5/9
        
        # Precipitation lags
        prcp_lag_1 = precipitation if not historical_precip or len(historical_precip) < 1 else historical_precip[-1] * 25.4
        prcp_lag_2 = precipitation if not historical_precip or len(historical_precip) < 2 else historical_precip[-2] * 25.4
        prcp_lag_3 = precipitation if not historical_precip or len(historical_precip) < 3 else historical_precip[-3] * 25.4
        prcp_lag_7 = precipitation if not historical_precip or len(historical_precip) < 7 else historical_precip[-7] * 25.4
        
        # Wind lags
        awnd_lag_1 = wind_speed if not historical_wind or len(historical_wind) < 1 else historical_wind[-1] * 1.609
        awnd_lag_2 = wind_speed if not historical_wind or len(historical_wind) < 2 else historical_wind[-2] * 1.609
        awnd_lag_3 = wind_speed if not historical_wind or len(historical_wind) < 3 else historical_wind[-3] * 1.609
        awnd_lag_7 = wind_speed if not historical_wind or len(historical_wind) < 7 else historical_wind[-7] * 1.609
        
        # VPD lags (simplified)
        vpd_lag_1 = vpd
        vpd_lag_2 = vpd
        vpd_lag_3 = vpd
        vpd_lag_7 = vpd
        
        # --- WEATHER ROLLING AVERAGES ---
        temp_roll_3 = temp_avg
        temp_roll_7 = temp_avg
        temp_roll_14 = temp_avg
        
        rain_roll_3 = precipitation * 3
        rain_roll_7 = precipitation * 7
        rain_roll_14 = precipitation * 14
        
        vpd_roll_3 = vpd
        vpd_roll_7 = vpd
        vpd_roll_14 = vpd
        
        # --- PEAK SEASON PROXIMITY ---
        tree_peak_prox = np.exp(-0.5 * ((day_of_year - 110) / 30) ** 2)  # DOY 110 ≈ mid-April
        grass_peak_prox = np.exp(-0.5 * ((day_of_year - 175) / 30) ** 2)  # DOY 175 ≈ late June
        weed_peak_prox = np.exp(-0.5 * ((day_of_year - 255) / 30) ** 2)  # DOY 255 ≈ mid-Sept
        
        # --- INTERACTIONS ---
        dry_x_wind = days_since_rain * wind_speed
        
        # Create comprehensive feature dictionary for weather-only model
        features = {
            # Advanced Biological
            'VPD': vpd,
            'Ventilation_Index': ventilation_index,
            'Shock_Index': shock_index,
            'GDD_Burst_5d': gdd_burst_5d,
            'Days_Since_Rain': days_since_rain,
            'Rain_Light': rain_light,
            'Rain_Heavy': rain_heavy,
            
            # Core Weather
            'TMAX': temp_max,
            'TMIN': temp_min,
            'TAVG': temp_avg,
            'PRCP': precipitation,
            'AWND': wind_speed,
            'Temp_Range': temp_range,
            'Is_Rainy': is_rainy,
            
            # Temporal
            'Day_of_Week': day_of_week,
            'Month_sin': month_sin,
            'Month_cos': month_cos,
            'DOY_sin': doy_sin,
            'DOY_cos': doy_cos,
            'Year_Numeric': year,
            'Month_Numeric': month,
            
            # Weather Lags
            'TMAX_lag_1': tmax_lag_1,
            'TMAX_lag_2': tmax_lag_2,
            'TMAX_lag_3': tmax_lag_3,
            'TMAX_lag_7': tmax_lag_7,
            'PRCP_lag_1': prcp_lag_1,
            'PRCP_lag_2': prcp_lag_2,
            'PRCP_lag_3': prcp_lag_3,
            'PRCP_lag_7': prcp_lag_7,
            'AWND_lag_1': awnd_lag_1,
            'AWND_lag_2': awnd_lag_2,
            'AWND_lag_3': awnd_lag_3,
            'AWND_lag_7': awnd_lag_7,
            'VPD_lag_1': vpd_lag_1,
            'VPD_lag_2': vpd_lag_2,
            'VPD_lag_3': vpd_lag_3,
            'VPD_lag_7': vpd_lag_7,
            
            # Weather Rolling
            'Temp_roll_3': temp_roll_3,
            'Temp_roll_7': temp_roll_7,
            'Temp_roll_14': temp_roll_14,
            'Rain_roll_3': rain_roll_3,
            'Rain_roll_7': rain_roll_7,
            'Rain_roll_14': rain_roll_14,
            'VPD_roll_3': vpd_roll_3,
            'VPD_roll_7': vpd_roll_7,
            'VPD_roll_14': vpd_roll_14,
            
            # GDD Features
            'GDD_General': gdd_general,
            'GDD_Tree': gdd_tree,
            'GDD_Grass': gdd_grass,
            'GDD_Weed': gdd_weed,
            'GDD_cumsum': gdd_cumsum,
            
            # Peak Proximity
            'Tree_Peak_Prox': tree_peak_prox,
            'Grass_Peak_Prox': grass_peak_prox,
            'Weed_Peak_Prox': weed_peak_prox,
            
            # Interactions
            'Dry_x_Wind': dry_x_wind,
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
        Predict daily pollen severity using weather-only trained models
        
        Returns:
            Tuple of (severity_score, severity_level)
        """
        if 'main' not in self.models:
            self.load_models()
        
        # Engineer features for weather-only model
        features_df = self._engineer_features(
            weather_data, date, historical_pollen,
            historical_temps, historical_precip, historical_wind
        )
        
        # Align features with model expectations
        if self.feature_names:
            # Create feature matrix with all expected features, fill missing with 0
            aligned_features = pd.DataFrame(0, index=features_df.index, columns=self.feature_names)
            
            # Fill in available features
            for col in features_df.columns:
                if col in aligned_features.columns:
                    aligned_features[col] = features_df[col]
            
            features_array = aligned_features.values
        else:
            # Fallback: use features as-is
            features_array = features_df.values
        
        # Log key features for debugging
        logger.info(f"Date: {date.date()}, DOY: {date.timetuple().tm_yday}")
        if 'VPD' in features_df.columns:
            logger.info(f"VPD: {features_df['VPD'].iloc[0]:.3f}, Ventilation: {features_df['Ventilation_Index'].iloc[0]:.2f}")
        logger.info(f"Peak prox: Tree={features_df['Tree_Peak_Prox'].iloc[0]:.3f}, Grass={features_df['Grass_Peak_Prox'].iloc[0]:.3f}, Weed={features_df['Weed_Peak_Prox'].iloc[0]:.3f}")
        
        # Make prediction using weather-only trained model
        model = self.models['main']
        severity = float(model.predict(features_array)[0])
        
        logger.info(f"Weather-only model prediction: {severity:.2f}")
        
        # Ensure within 0-10 range
        severity = max(0.0, min(10.0, severity))
        level = self._severity_to_level(severity)
        
        return severity, level
    
    def predict_weekly(self, weather_forecast: List[Dict], 
                      current_pollen: Optional[float] = None) -> List[Tuple[datetime, float, str]]:
        """
        Predict weekly pollen forecast using weather-only models
        
        Returns:
            List of (date, severity, level) tuples
        """
        predictions = []
        
        # For weather-only models, we don't need pollen history
        # Historical weather data will be built up as we make predictions
        historical_temps = []
        historical_precip = []
        historical_wind = []
        
        for i, day_weather in enumerate(weather_forecast):
            date = day_weather['date']
            weather = {k: v for k, v in day_weather.items() if k != 'date'}
            
            # Use accumulated weather history for better predictions
            severity, level = self.predict_daily(
                weather, date, 
                historical_pollen=None,  # Weather-only models don't need pollen history
                historical_temps=historical_temps[-30:] if historical_temps else None,
                historical_precip=historical_precip[-90:] if historical_precip else None,
                historical_wind=historical_wind[-30:] if historical_wind else None
            )
            predictions.append((date, severity, level))
            
            # Build up weather history for subsequent predictions
            historical_temps.append(weather.get('temp_avg', (weather.get('temp_max', 70) + weather.get('temp_min', 50)) / 2))
            historical_precip.append(weather.get('precipitation', 0.0))
            historical_wind.append(weather.get('wind_speed', 5.0))
            
            logger.info(f"Weekly forecast day {i+1}: {date.date()} -> {severity:.2f} ({level})")
        
        return predictions
    
    def identify_allergens(self, weather_data: Dict, date: datetime,
                          historical_pollen: Optional[List[float]] = None,
                          historical_temps: Optional[List[float]] = None,
                          historical_precip: Optional[List[float]] = None,
                          historical_wind: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Identify allergen contributions using weather-only trained allergen-specific models
        
        Returns:
            Dictionary of allergen types to severity scores (0-10)
        """
        if 'main' not in self.models:
            self.load_models()
            
        # Check if we have weather-only allergen models loaded
        allergen_models_available = all(k in self.models for k in ['grass', 'tree', 'ragweed', 'weed'])
        
        if not allergen_models_available:
            # Fallback: estimate from main model using seasonal patterns
            logger.warning("Weather-only allergen models not available, using seasonal estimation")
            severity, _ = self.predict_daily(
                weather_data, date, historical_pollen, 
                historical_temps, historical_precip, historical_wind
            )
            
            # Estimate allergen distribution based on season and weather conditions
            month = date.month
            temp_avg_f = weather_data.get('temp_avg', 65.0)
            
            # Adjust seasonal patterns based on temperature
            temp_factor = 1.0
            if temp_avg_f > 75:  # Hot weather boosts grass/weed
                temp_factor = 1.2
            elif temp_avg_f < 50:  # Cold weather reduces all
                temp_factor = 0.7
            
            if month in [3, 4, 5]:  # Spring - Tree dominant
                return {
                    'tree': severity * 0.6 * temp_factor, 
                    'grass': severity * 0.2 * temp_factor, 
                    'ragweed': severity * 0.1 * temp_factor, 
                    'weed': severity * 0.1 * temp_factor
                }
            elif month in [6, 7, 8]:  # Summer - Grass dominant
                return {
                    'tree': severity * 0.1 * temp_factor, 
                    'grass': severity * 0.7 * temp_factor,
                    'ragweed': severity * 0.1 * temp_factor, 
                    'weed': severity * 0.1 * temp_factor
                }
            elif month in [9, 10]:  # Fall - Ragweed/Weed dominant
                return {
                    'tree': severity * 0.1 * temp_factor, 
                    'grass': severity * 0.2 * temp_factor,
                    'ragweed': severity * 0.5 * temp_factor, 
                    'weed': severity * 0.2 * temp_factor
                }
            else:  # Winter - Low activity
                return {
                    'tree': severity * 0.3 * temp_factor, 
                    'grass': severity * 0.1 * temp_factor,
                    'ragweed': severity * 0.1 * temp_factor, 
                    'weed': severity * 0.5 * temp_factor
                }
        
        # Use weather-only trained allergen-specific models
        logger.info(f"Using weather-only allergen models for {date.date()}")
        features_df = self._engineer_features(
            weather_data, date, historical_pollen,
            historical_temps, historical_precip, historical_wind
        )
        
        # Align features for each allergen model
        allergen_scores = {}
        for allergen in ['tree', 'grass', 'weed', 'ragweed']:
            if allergen in self.models:
                # Align features with model expectations
                if self.feature_names:
                    aligned_features = pd.DataFrame(0, index=features_df.index, columns=self.feature_names)
                    for col in features_df.columns:
                        if col in aligned_features.columns:
                            aligned_features[col] = features_df[col]
                    features_array = aligned_features.values
                else:
                    features_array = features_df.values
                
                # Weather-only models predict severity (0-10) directly
                score = float(self.models[allergen].predict(features_array)[0])
                allergen_scores[allergen] = max(0.0, min(10.0, score))
                logger.info(f"Weather-only {allergen.capitalize()} severity: {allergen_scores[allergen]:.2f}")
        
        return allergen_scores


# Global service instance
prediction_service = PredictionService()
