#!/usr/bin/env python3
"""
Simple test of the pollen predictor with sample data
"""

import numpy as np
import joblib
from datetime import datetime

# Test loading the model
try:
    model = joblib.load('pollen_predictor_random_forest.joblib')
    feature_cols = joblib.load('model_features.joblib')
    print("✅ Model loaded successfully!")
    print(f"✅ Model expects {len(feature_cols)} features")
    print("\nFirst 10 expected features:")
    for i, feature in enumerate(feature_cols[:10]):
        print(f"  {i+1}. {feature}")
    
    # Create a sample prediction with default values
    print("\n🧪 Testing with sample spring day...")
    
    # Sample feature vector (all zeros as baseline)
    sample_features = np.zeros(len(feature_cols))
    
    # Set some realistic values for key features
    feature_dict = {
        'TMAX': 20.0,          # 20°C max temp
        'TMIN': 8.0,           # 8°C min temp  
        'TAVG': 14.0,          # 14°C average
        'PRCP': 0.0,           # No precipitation
        'AWND': 5.0,           # 5 m/s wind
        'Month': 5,            # May
        'Day_of_Year': 125,    # Early May
        'Temp_Range': 12.0,    # 12°C temperature range
        'Month_sin': np.sin(2 * np.pi * 5 / 12),
        'Month_cos': np.cos(2 * np.pi * 5 / 12),
        'Pollen_roll_mean_3': 150,  # Moderate recent pollen
    }
    
    # Fill in the feature vector
    for i, feature_name in enumerate(feature_cols):
        if feature_name in feature_dict:
            sample_features[i] = feature_dict[feature_name]
    
    # Make prediction
    X = sample_features.reshape(1, -1)
    prediction = model.predict(X)[0]
    
    print(f"\n🌸 Sample Prediction:")
    print(f"   Input: Spring day, 20°C max, 8°C min, no rain, moderate wind")
    print(f"   Predicted Pollen Severity: {prediction:.2f}/10")
    
    # Interpret result
    if prediction <= 2:
        level = "Low"
    elif prediction <= 5:
        level = "Moderate" 
    elif prediction <= 7:
        level = "High"
    else:
        level = "Very High"
    
    print(f"   Severity Level: {level}")
    print("\n✅ Model is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()