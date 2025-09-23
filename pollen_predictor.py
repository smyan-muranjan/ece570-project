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
        """Initialize the predictor with trained model and feature engineering"""
        try:
            self.model = joblib.load('pollen_intensity_model.pkl')
            self.scaler = joblib.load('pollen_scaler.pkl')
            print("✅ Model loaded successfully!")
        except FileNotError:
            print("❌ Model files not found. Please run the training script first.")
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
        for days_ago in [1, 2, 3]:
            while True:
                pollen_input = input(f"Pollen level {days_ago} day(s) ago (0-10, or press Enter to skip): ").strip()
                if not pollen_input:
                    recent_pollen.append(None)
                    break
                try:
                    pollen_val = float(pollen_input)
                    if 0 <= pollen_val <= 10:
                        recent_pollen.append(pollen_val)
                        break
                    else:
                        print("❌ Pollen level must be between 0 and 10")
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
        """Engineer features from user input"""
        date = user_data['date']
        
        # Basic features
        features = {
            'TMAX': user_data['tmax'],
            'TMIN': user_data['tmin'],
            'TAVG': (user_data['tmax'] + user_data['tmin']) / 2,
            'PRCP': user_data['prcp'],
            'AWND': user_data['wind'],
            
            # Date features
            'Year': date.year,
            'Month': date.month,
            'Day': date.day,
            'Day_of_Year': date.timetuple().tm_yday,
            'Day_of_Week': date.weekday(),
            
            # Temperature derived features
            'Temperature_Range': user_data['tmax'] - user_data['tmin'],
            'GDD_Base10': max(0, features['TAVG'] - 10) if 'TAVG' in locals() else max(0, (user_data['tmax'] + user_data['tmin']) / 2 - 10),
            
            # Seasonal features
            'Spring': 1 if 3 <= date.month <= 5 else 0,
            'Summer': 1 if 6 <= date.month <= 8 else 0,
            'Fall': 1 if 9 <= date.month <= 11 else 0,
            'Winter': 1 if date.month in [12, 1, 2] else 0,
            
            # Precipitation features
            'Rain_Day': 1 if user_data['prcp'] > 1 else 0,
            'Heavy_Rain': 1 if user_data['prcp'] > 10 else 0,
        }
        
        # Add trigonometric features for seasonality
        features['Month_Sin'] = np.sin(2 * np.pi * date.month / 12)
        features['Month_Cos'] = np.cos(2 * np.pi * date.month / 12)
        features['Day_Sin'] = np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        features['Day_Cos'] = np.cos(2 * np.pi * date.timetuple().tm_yday / 365)
        
        # Handle recent pollen data
        for i, pollen_val in enumerate(user_data['recent_pollen']):
            if pollen_val is not None:
                features[f'Pollen_Lag_{i+1}'] = pollen_val
                features[f'Pollen_Rolling_Mean_{i+1}'] = np.mean([p for p in user_data['recent_pollen'][:i+1] if p is not None])
            else:
                # Use seasonal averages as fallback
                seasonal_avg = self.get_seasonal_average(date.month)
                features[f'Pollen_Lag_{i+1}'] = seasonal_avg
                features[f'Pollen_Rolling_Mean_{i+1}'] = seasonal_avg
        
        return features
    
    def get_seasonal_average(self, month):
        """Get approximate seasonal average pollen levels"""
        seasonal_averages = {
            1: 0.5, 2: 0.8, 3: 2.0, 4: 6.5, 5: 7.2, 6: 4.8,
            7: 3.5, 8: 3.2, 9: 4.0, 10: 2.8, 11: 1.2, 12: 0.6
        }
        return seasonal_averages.get(month, 2.0)
    
    def predict_pollen(self, user_data):
        """Make pollen prediction from user data"""
        # Engineer features
        features = self.engineer_features(user_data)
        
        # Create feature vector (ensure all required features are present)
        # This should match the features used during training
        required_features = [
            'TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND', 'Year', 'Month', 'Day',
            'Day_of_Year', 'Day_of_Week', 'Temperature_Range', 'GDD_Base10',
            'Spring', 'Summer', 'Fall', 'Winter', 'Rain_Day', 'Heavy_Rain',
            'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos',
            'Pollen_Lag_1', 'Pollen_Lag_2', 'Pollen_Lag_3',
            'Pollen_Rolling_Mean_1', 'Pollen_Rolling_Mean_2', 'Pollen_Rolling_Mean_3'
        ]
        
        # Create feature vector
        feature_vector = []
        for feature in required_features:
            if feature in features:
                feature_vector.append(features[feature])
            else:
                feature_vector.append(0)  # Default value for missing features
        
        # Convert to numpy array and reshape
        X = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Ensure prediction is within valid range
        prediction = max(0, min(10, prediction))
        
        return prediction, features
    
    def interpret_prediction(self, prediction, user_data):
        """Provide interpretation and recommendations"""
        print(f"\n🌸 POLLEN FORECAST RESULTS")
        print("=" * 50)
        
        # Severity level
        if prediction <= 2:
            level = "Very Low"
            color = "🟢"
            advice = "Great day to be outside! Minimal allergy risk."
        elif prediction <= 4:
            level = "Low" 
            color = "🟡"
            advice = "Good day for outdoor activities. Sensitive individuals may notice mild symptoms."
        elif prediction <= 6:
            level = "Moderate"
            color = "🟠"
            advice = "Consider taking allergy medication. Limit prolonged outdoor exposure."
        elif prediction <= 8:
            level = "High"
            color = "🔴"
            advice = "Take allergy medication. Limit outdoor time, especially in morning."
        else:
            level = "Very High"
            color = "🟣"
            advice = "Stay indoors if possible. Keep windows closed. Take allergy medication."
        
        print(f"📊 Predicted Pollen Level: {prediction:.1f}/10")
        print(f"{color} Severity: {level}")
        print(f"💡 Recommendation: {advice}")
        
        # Additional insights
        print(f"\n🔍 WEATHER ANALYSIS:")
        print(f"   Temperature: {user_data['tmin']:.1f}°C - {user_data['tmax']:.1f}°C")
        print(f"   Precipitation: {user_data['prcp']:.1f}mm")
        print(f"   Wind Speed: {user_data['wind']:.1f} m/s")
        
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
        
        # Weather impact
        impacts = []
        if user_data['prcp'] > 5:
            impacts.append("Heavy rain will wash away pollen (lower risk)")
        elif user_data['prcp'] > 1:
            impacts.append("Light rain may reduce pollen levels")
        
        if user_data['wind'] > 15:
            impacts.append("High winds may spread pollen widely")
        elif user_data['wind'] < 5:
            impacts.append("Low winds may cause pollen to accumulate locally")
        
        temp_avg = (user_data['tmax'] + user_data['tmin']) / 2
        if temp_avg > 25:
            impacts.append("Warm temperatures promote pollen release")
        elif temp_avg < 10:
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

if __name__ == "__main__":
    main()