#!/usr/bin/env python3
"""
Interactive Pollen Severity Predictor
Allows users to input current weather conditions and get personalized pollen forecasts
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PollenPredictor:
    def __init__(self):
        """Initialize the predictor with trained model and features"""
        try:
            # Load the actual model files we created
            self.model = joblib.load('pollen_predictor_random_forest.joblib')
            self.feature_cols = joblib.load('model_features.joblib')
            print("✅ Model loaded successfully!")
            print(f"✅ Features loaded: {len(self.feature_cols)} features")
        except FileNotFoundError:
            print("❌ Model files not found. Please run pollen_intensity_predictor.py first.")
            exit(1)
    
    def get_user_input(self):
        """Get weather and time information from user"""
        print("\n🌸 POLLEN SEVERITY PREDICTOR")
        print("=" * 50)
        print("Enter current weather conditions to get your pollen forecast:")
        
        # Get date information
        print("\n📅 DATE INFORMATION:")
        use_today = input("Use today's date? (y/n): ").lower().strip() == 'y'
        
        if use_today:
            date = datetime.now()
            print(f"Using today: {date.strftime('%Y-%m-%d')}")
        else:
            while True:
                try:
                    date_str = input("Enter date (YYYY-MM-DD): ")
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    break
                except ValueError:
                    print("❌ Invalid date format. Please use YYYY-MM-DD")
        
        # Get weather information
        print("\n🌤️ WEATHER CONDITIONS:")
        
        # Temperature
        while True:
            try:
                temp_unit = input("Temperature unit - (F)ahrenheit or (C)elsius? [F]: ").lower().strip()
                if not temp_unit:
                    temp_unit = 'f'
                
                if temp_unit not in ['f', 'c']:
                    print("❌ Please enter 'F' or 'C'")
                    continue
                
                tmax = float(input("Maximum temperature today: "))
                tmin = float(input("Minimum temperature today: "))
                
                # Convert to Celsius if needed
                if temp_unit == 'f':
                    tmax = (tmax - 32) * 5/9
                    tmin = (tmin - 32) * 5/9
                
                if tmin > tmax:
                    print("❌ Minimum temperature cannot be higher than maximum")
                    continue
                
                break
            except ValueError:
                print("❌ Please enter valid numbers for temperature")
        
        # Precipitation
        while True:
            try:
                precip_unit = input("Precipitation unit - (mm) or (inches)? [mm]: ").lower().strip()
                if not precip_unit:
                    precip_unit = 'mm'
                
                if precip_unit not in ['mm', 'inches', 'in']:
                    print("❌ Please enter 'mm' or 'inches'")
                    continue
                
                prcp = float(input("Precipitation amount (0 if none): "))
                
                # Convert to mm if needed
                if precip_unit in ['inches', 'in']:
                    prcp = prcp * 25.4
                
                if prcp < 0:
                    print("❌ Precipitation cannot be negative")
                    continue
                
                break
            except ValueError:
                print("❌ Please enter a valid number for precipitation")
        
        # Wind speed
        while True:
            try:
                wind_unit = input("Wind speed unit - (mph) or (m/s)? [mph]: ").lower().strip()
                if not wind_unit:
                    wind_unit = 'mph'
                
                if wind_unit not in ['mph', 'm/s', 'ms']:
                    print("❌ Please enter 'mph' or 'm/s'")
                    continue
                
                wind = float(input("Average wind speed: "))
                
                # Convert to m/s if needed
                if wind_unit == 'mph':
                    wind = wind * 0.44704
                
                if wind < 0:
                    print("❌ Wind speed cannot be negative")
                    continue
                
                break
            except ValueError:
                print("❌ Please enter a valid number for wind speed")
        
        # Optional: Recent pollen levels
        print("\n🌸 RECENT POLLEN HISTORY (optional):")
        print("If you know recent pollen levels, this improves accuracy:")
        
        recent_pollen = []
        for days_ago in [1, 3, 7]:
            while True:
                pollen_input = input(f"Pollen level {days_ago} day(s) ago (0-500+ count, or press Enter to skip): ").strip()
                if not pollen_input:
                    recent_pollen.append(None)
                    break
                try:
                    pollen_val = float(pollen_input)
                    if pollen_val >= 0:
                        recent_pollen.append(pollen_val)
                        break
                    else:
                        print("❌ Pollen level must be non-negative")
                except ValueError:
                    print("❌ Please enter a valid number")
        
        return {
            'date': date,
            'tmax': tmax,
            'tmin': tmin,
            'prcp': prcp,
            'wind': wind,
            'recent_pollen': recent_pollen
        }
    
    def engineer_features(self, user_data):
        """Engineer features that match our trained model"""
        date = user_data['date']
        
        # Create base features dictionary
        features = {}
        
        # Weather features
        features['TMAX'] = user_data['tmax']
        features['TMIN'] = user_data['tmin']
        features['TAVG'] = (user_data['tmax'] + user_data['tmin']) / 2
        features['PRCP'] = user_data['prcp']
        features['AWND'] = user_data['wind']
        
        # Temporal features
        features['Month'] = date.month
        features['Day_of_Year'] = date.timetuple().tm_yday
        features['Day_of_Week'] = date.weekday()
        features['Year_Numeric'] = date.year
        features['Month_Numeric'] = date.month
        
        # Cyclical encoding
        features['Month_sin'] = np.sin(2 * np.pi * date.month / 12)
        features['Month_cos'] = np.cos(2 * np.pi * date.month / 12)
        features['Day_of_Year_sin'] = np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        features['Day_of_Year_cos'] = np.cos(2 * np.pi * date.timetuple().tm_yday / 365)
        
        # Weather derived features
        features['Temp_Range'] = user_data['tmax'] - user_data['tmin']
        features['Is_Rainy'] = 1 if user_data['prcp'] > 0 else 0
        features['High_Wind'] = 1 if user_data['wind'] > 7.5 else 0  # Approximate 75th percentile
        
        # Growing degree days
        features['GDD'] = max(0, features['TAVG'] - 5)
        # Estimate cumulative GDD based on day of year and typical patterns
        if date.month <= 3:
            features['GDD_cumsum'] = features['GDD'] * date.timetuple().tm_yday * 0.1
        else:
            features['GDD_cumsum'] = features['GDD'] * date.timetuple().tm_yday * 0.3
        
        # Handle lag features with user input or seasonal estimates
        seasonal_avg = self.get_seasonal_average(date.month)
        
        # Lag features
        for i, lag in enumerate([1, 3, 7]):
            if i < len(user_data['recent_pollen']) and user_data['recent_pollen'][i] is not None:
                features[f'Pollen_lag_{lag}'] = user_data['recent_pollen'][i]
                features[f'TMAX_lag_{lag}'] = user_data['tmax']  # Assume similar weather
                features[f'PRCP_lag_{lag}'] = user_data['prcp'] * 0.5  # Estimate
            else:
                features[f'Pollen_lag_{lag}'] = seasonal_avg
                features[f'TMAX_lag_{lag}'] = user_data['tmax']
                features[f'PRCP_lag_{lag}'] = 0
        
        # Rolling averages
        recent_pollen_values = [p if p is not None else seasonal_avg for p in user_data['recent_pollen']]
        while len(recent_pollen_values) < 3:
            recent_pollen_values.append(seasonal_avg)
        
        for window in [3, 7, 14]:
            features[f'Pollen_roll_mean_{window}'] = np.mean(recent_pollen_values[:min(window, len(recent_pollen_values))])
            features[f'Temp_roll_mean_{window}'] = user_data['tmax']  # Simplification
            features[f'Precip_roll_sum_{window}'] = user_data['prcp'] * min(window, 3)
        
        return features
    
    def get_seasonal_average(self, month):
        """Get approximate seasonal average pollen levels based on our data"""
        seasonal_averages = {
            1: 5, 2: 10, 3: 25, 4: 145, 5: 380, 6: 200,
            7: 8, 8: 25, 9: 12, 10: 2, 11: 1, 12: 3
        }
        return seasonal_averages.get(month, 50)
    
    def predict_pollen(self, user_data):
        """Make pollen prediction from user data"""
        # Engineer features
        features = self.engineer_features(user_data)
        
        # Create feature vector matching the trained model
        feature_vector = []
        for feature_name in self.feature_cols:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                # Provide reasonable defaults for missing features
                if 'lag' in feature_name:
                    feature_vector.append(self.get_seasonal_average(user_data['date'].month))
                elif 'roll' in feature_name:
                    feature_vector.append(self.get_seasonal_average(user_data['date'].month))
                elif '_sin' in feature_name or '_cos' in feature_name:
                    feature_vector.append(0)
                else:
                    feature_vector.append(0)
        
        # Convert to numpy array and reshape
        X = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Ensure prediction is within valid range (0-10)
        prediction = max(0, min(10, prediction))
        
        return prediction, features
    
    def interpret_prediction(self, prediction, user_data):
        """Provide interpretation and recommendations"""
        print(f"\n🌸 POLLEN FORECAST RESULTS")
        print("=" * 50)
        
        # Severity level and advice based on 0-10 scale
        if prediction <= 1:
            level = "None to Very Low"
            color = "🟢"
            advice = "Excellent day to be outside! Minimal allergy risk."
        elif prediction <= 3:
            level = "Low"
            color = "🟡"
            advice = "Good day for outdoor activities. Sensitive individuals may notice mild symptoms."
        elif prediction <= 5:
            level = "Moderate"
            color = "🟠"
            advice = "Consider taking allergy medication. Limit prolonged outdoor exposure."
        elif prediction <= 7:
            level = "High"
            color = "🔴"
            advice = "Take allergy medication. Limit outdoor time, especially in morning."
        else:
            level = "Very High"
            color = "🟣"
            advice = "Stay indoors if possible. Keep windows closed. Take allergy medication."
        
        print(f"📊 Predicted Pollen Severity: {prediction:.1f}/10")
        print(f"{color} Severity Level: {level}")
        print(f"💡 Recommendation: {advice}")
        
        # Additional insights
        print(f"\n🔍 WEATHER ANALYSIS:")
        temp_f_max = user_data['tmax'] * 9/5 + 32
        temp_f_min = user_data['tmin'] * 9/5 + 32
        prcp_in = user_data['prcp'] / 25.4
        wind_mph = user_data['wind'] / 0.44704
        
        print(f"   Temperature: {temp_f_min:.1f}°F - {temp_f_max:.1f}°F ({user_data['tmin']:.1f}°C - {user_data['tmax']:.1f}°C)")
        print(f"   Precipitation: {prcp_in:.2f}\" ({user_data['prcp']:.1f}mm)")
        print(f"   Wind Speed: {wind_mph:.1f} mph ({user_data['wind']:.1f} m/s)")
        
        # Seasonal context
        month = user_data['date'].month
        if 3 <= month <= 5:
            season_note = "🌸 Spring: Tree pollen season - expect higher levels"
        elif 6 <= month <= 8:
            season_note = "☀️ Summer: Grass pollen dominant"
        elif 9 <= month <= 11:
            season_note = "🍂 Fall: Ragweed season may affect sensitive individuals"
        else:
            season_note = "❄️ Winter: Typically lowest pollen levels"
        
        print(f"   Season: {season_note}")
        
        # Weather impact analysis
        impacts = []
        if user_data['prcp'] > 5:
            impacts.append("Rain will wash away pollen (lower risk)")
        elif user_data['prcp'] > 1:
            impacts.append("Light rain may reduce pollen levels")
        
        if user_data['wind'] > 10:
            impacts.append("High winds may spread pollen widely")
        elif user_data['wind'] < 3:
            impacts.append("Low winds may cause pollen to accumulate locally")
        
        temp_avg = (user_data['tmax'] + user_data['tmin']) / 2
        if temp_avg > 25:
            impacts.append("Warm temperatures promote pollen release")
        elif temp_avg < 5:
            impacts.append("Cool temperatures reduce pollen activity")
        
        if impacts:
            print(f"\n🌤️ WEATHER IMPACT:")
            for impact in impacts:
                print(f"   • {impact}")

def main():
    """Main interactive function"""
    print("🌸 Welcome to the Pollen Severity Predictor!")
    print("This tool uses machine learning to predict daily pollen levels")
    print("based on weather conditions and seasonal patterns.\n")
    
    try:
        predictor = PollenPredictor()
        
        while True:
            try:
                # Get user input
                user_data = predictor.get_user_input()
                
                # Make prediction
                prediction, features = predictor.predict_pollen(user_data)
                
                # Show results
                predictor.interpret_prediction(prediction, user_data)
                
                # Ask for another prediction
                print(f"\n" + "="*50)
                another = input("Would you like another prediction? (y/n): ").lower().strip()
                if another != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! Stay safe during allergy season! 🌸")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please try again.")
                
    except SystemExit:
        pass  # Model loading failed, already handled

if __name__ == "__main__":
    main()