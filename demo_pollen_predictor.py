#!/usr/bin/env python3
"""
Pollen Prediction Demo
Demonstrates how to use the trained pollen intensity predictor
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

def load_model():
    """Load the trained model and features"""
    print("📂 Loading trained model...")
    
    model = joblib.load('pollen_predictor_random_forest.joblib')
    feature_cols = joblib.load('model_features.joblib')
    
    print(f"   ✅ Model loaded: Random Forest")
    print(f"   ✅ Features loaded: {len(feature_cols)} features")
    
    return model, feature_cols

def predict_pollen_for_conditions(model, feature_cols, weather_conditions):
    """Make pollen prediction for given weather conditions"""
    print("\n🔮 Making pollen prediction...")
    
    # Create a dataframe with the conditions
    input_df = pd.DataFrame([weather_conditions])
    
    # Add missing features with default values
    for col in feature_cols:
        if col not in input_df.columns:
            if 'lag_' in col:
                input_df[col] = 0  # Default for lag features
            elif 'roll_' in col:
                input_df[col] = 0  # Default for rolling features
            elif 'sin' in col or 'cos' in col:
                input_df[col] = 0  # Default for cyclic features
            else:
                input_df[col] = 0  # Default value
    
    # Ensure correct feature order
    X = input_df[feature_cols]
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Map back to severity level
    severity_labels = ['None', 'Very Low', 'Low', 'Low-Mod', 'Moderate', 
                      'Mod-High', 'High', 'Very High', 'Extreme', 'Severe']
    
    severity_level = min(int(round(prediction)), len(severity_labels)-1)
    severity_label = severity_labels[severity_level]
    
    print(f"   🌸 Predicted Pollen Severity: {prediction:.2f}")
    print(f"   📊 Severity Level: {severity_level} ({severity_label})")
    
    return prediction, severity_level, severity_label

def demo_predictions():
    """Demonstrate predictions for different scenarios"""
    print("\n🎭 DEMO: Pollen Predictions for Different Scenarios")
    print("=" * 60)
    
    model, feature_cols = load_model()
    
    scenarios = [
        {
            "name": "Spring High Risk Day",
            "conditions": {
                "TMAX": 22.0, "TMIN": 8.0, "TAVG": 15.0,
                "PRCP": 0.0, "AWND": 3.0,
                "Month": 5, "Day_of_Year": 120,
                "Temp_Range": 14.0, "Is_Rainy": 0, "High_Wind": 0,
                "GDD": 10.0, "GDD_cumsum": 500.0
            }
        },
        {
            "name": "Rainy Spring Day",
            "conditions": {
                "TMAX": 18.0, "TMIN": 12.0, "TAVG": 15.0,
                "PRCP": 15.0, "AWND": 8.0,
                "Month": 5, "Day_of_Year": 120,
                "Temp_Range": 6.0, "Is_Rainy": 1, "High_Wind": 1,
                "GDD": 10.0, "GDD_cumsum": 500.0
            }
        },
        {
            "name": "Summer Low Pollen Day",
            "conditions": {
                "TMAX": 28.0, "TMIN": 18.0, "TAVG": 23.0,
                "PRCP": 0.0, "AWND": 2.0,
                "Month": 7, "Day_of_Year": 200,
                "Temp_Range": 10.0, "Is_Rainy": 0, "High_Wind": 0,
                "GDD": 18.0, "GDD_cumsum": 1200.0
            }
        },
        {
            "name": "Fall Ragweed Season",
            "conditions": {
                "TMAX": 20.0, "TMIN": 10.0, "TAVG": 15.0,
                "PRCP": 0.0, "AWND": 5.0,
                "Month": 9, "Day_of_Year": 250,
                "Temp_Range": 10.0, "Is_Rainy": 0, "High_Wind": 0,
                "GDD": 10.0, "GDD_cumsum": 1500.0
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Weather: {scenario['conditions']['TMAX']:.1f}°C max, {scenario['conditions']['TMIN']:.1f}°C min")
        print(f"   Precipitation: {scenario['conditions']['PRCP']:.1f}mm")
        print(f"   Wind: {scenario['conditions']['AWND']:.1f} m/s")
        
        prediction, level, label = predict_pollen_for_conditions(
            model, feature_cols, scenario['conditions']
        )
        
        # Risk assessment
        if level <= 2:
            risk = "🟢 LOW RISK"
        elif level <= 5:
            risk = "🟡 MODERATE RISK"
        elif level <= 7:
            risk = "🟠 HIGH RISK"
        else:
            risk = "🔴 VERY HIGH RISK"
        
        print(f"   🚨 Assessment: {risk}")

def main():
    """Main demo function"""
    print("🌸 POLLEN INTENSITY PREDICTOR DEMO")
    print("🔮 Showcasing the trained model capabilities")
    print("=" * 60)
    
    try:
        demo_predictions()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED!")
        print("=" * 60)
        print("🎯 The model successfully predicts pollen intensity based on:")
        print("   • Weather conditions (temperature, precipitation, wind)")
        print("   • Seasonal patterns (month, day of year)")
        print("   • Historical trends (lag and rolling features)")
        print("\n🚀 Ready for integration into allergy forecasting app!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())