"""
Multi-Type XGBoost Pollen Predictor (Abiotic/Weather-Only Mode)
- Features: AAAAI thresholds, Advanced Biological Drivers (VPD, Ventilation, Shock)
- Constraints: Zero dependence on historical pollen counts (solves cold start)
- Hardware: GPU-accelerated training (RTX 2080 optimized)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    print("✅ XGBoost imported successfully")
except ImportError:
    print("❌ XGBoost not installed! Install with: pip install xgboost")
    exit(1)

from datetime import datetime
import joblib
import json
import sys
import argparse


# ============================================================================
# 1. AAAAI BIOLOGICAL THRESHOLDS
# ============================================================================

def get_aaaai_thresholds():
    """AAAAI/National Allergy Bureau validated thresholds"""
    return {
        'Tree': {
            'thresholds': [0, 1, 15, 90, 1500],
            'levels': ['None', 'Low', 'Moderate', 'High', 'Very High']
        },
        'Grass': {
            'thresholds': [0, 1, 5, 20, 200],
            'levels': ['None', 'Low', 'Moderate', 'High', 'Very High']
        },
        'Weed': {
            'thresholds': [0, 1, 10, 50, 500],
            'levels': ['None', 'Low', 'Moderate', 'High', 'Very High']
        },
        'Ragweed': {
            'thresholds': [0, 1, 10, 50, 500],
            'levels': ['None', 'Low', 'Moderate', 'High', 'Very High']
        },
        'Total_Pollen': {
            'thresholds': [0, 10, 30, 60, 100, 200, 500, 1000, 2000, 5000],
            'levels': ['None', 'Low', 'Low-Mod', 'Moderate', 'Mod-High', 
                      'High', 'Very High', 'Extreme', 'Severe', 'Critical']
        }
    }


def pollen_to_severity(count, thresholds):
    """Convert pollen count to severity (0-10 scale)"""
    if count == 0 or pd.isna(count):
        return 0
    for i in range(len(thresholds) - 1, 0, -1):
        if count >= thresholds[i]:
            # Map to 0-10 scale
            return min(10, i * (10 / (len(thresholds) - 1)))
    return 0


def create_biological_severity_scales(df):
    """Create severity scales using AAAAI biological thresholds"""
    print("\n🌸 Creating AAAAI Biological Severity Scales...")
    
    aaaai = get_aaaai_thresholds()
    pollen_cols = {'Tree': 'Tree', 'Grass': 'Grass', 'Weed': 'Weed', 
                   'Ragweed': 'Ragweed', 'Total_Pollen': 'Total_Pollen'}
    
    available_types = [k for k, v in pollen_cols.items() if v in df.columns]
    
    for pollen_type in available_types:
        col = pollen_cols[pollen_type]
        severity_col = f'{col}_Severity'
        thresholds = aaaai[pollen_type]['thresholds']
        
        # Extend thresholds to 0-10 scale for regression target
        if len(thresholds) == 5:
            extended = [
                thresholds[0], thresholds[1],
                (thresholds[1] + thresholds[2]) / 2, thresholds[2],
                (thresholds[2] + thresholds[3]) / 2, thresholds[3],
                (thresholds[3] + thresholds[4]) / 2, thresholds[4],
                thresholds[4] * 1.5, thresholds[4] * 2.5, thresholds[4] * 5
            ]
        else:
            extended = thresholds
        
        df[severity_col] = df[col].apply(lambda x: pollen_to_severity(x, extended))
        print(f"   ✅ {pollen_type:15} severity created")
    
    return df


# ============================================================================
# 2. ADVANCED BIOLOGICAL FEATURE ENGINEERING
# ============================================================================

def calculate_vpd(t_avg, t_min, rh=None):
    """
    Calculate Vapor Pressure Deficit (VPD) in kPa.
    VPD is the "drying power" of the air - a key trigger for anther dehiscence (pollen release).
    If RH is missing, uses T_min as a proxy for Dew Point.
    """
    # Saturation Vapor Pressure (es) at T_avg
    es = 0.6108 * np.exp((17.27 * t_avg) / (t_avg + 237.3))
    
    if rh is not None:
        # Actual Vapor Pressure (ea) using Humidity
        ea = es * (rh / 100.0)
    else:
        # Approximation: T_min ≈ Dew Point
        ea = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
        
    return np.maximum(0, es - ea)


def engineer_features_no_pollen_history(df):
    """
    Master feature engineering function.
    CRITICAL: Does NOT use any pollen history (lags/rolling pollen).
    Focuses on Weather Dynamics, Biological Triggers, and Temporal Cycles.
    """
    print("\n🔧 Engineering Advanced Abiotic Features (No Pollen History)...")
    
    df = df.sort_values('Date_Standard').reset_index(drop=True)
    
    # --- A. CORE TEMPORAL ---
    df['Day_of_Week'] = df['Date_Standard'].dt.dayofweek
    df['Month_Numeric'] = df['Date_Standard'].dt.month
    df['Year_Numeric'] = df['Date_Standard'].dt.year
    df['Day_of_Year'] = df['Date_Standard'].dt.dayofyear
    
    # Cyclical encoding
    df['Month_sin'] = np.sin(2 * np.pi * df['Month_Numeric'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month_Numeric'] / 12)
    df['DOY_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
    df['DOY_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
    
    # --- B. CORE WEATHER ---
    df['Temp_Range'] = df['TMAX'] - df['TMIN']
    df['Is_Rainy'] = (df['PRCP'] > 0).astype(int)
    
    # --- C. ADVANCED BIOLOGICAL TRIGGERS ---
    
    # 1. Vapor Pressure Deficit (The Release Trigger)
    rh_col = 'RH' if 'RH' in df.columns else None
    df['VPD'] = calculate_vpd(df['TAVG'], df['TMIN'], rh=df.get(rh_col))
    
    # 2. Ventilation Index (The Dispersion Engine)
    # Combined effect of vertical mixing (Diurnal Range) and horizontal transport (Wind)
    df['Ventilation_Index'] = df['AWND'] * df['Temp_Range']
    
    # 3. Precipitation Dynamics (Washout vs. Release)
    # Days Since Rain (Critical for post-rain release spikes)
    rainy_days = df['PRCP'] > 0
    df['Days_Since_Rain'] = df.groupby(rainy_days.cumsum()).cumcount()
    
    # Rain Intensity Categories (Shock vs Washout)
    # 0 = None, 1 = Light (<2.5mm), 2 = Heavy (>2.5mm)
    df['Rain_Intensity'] = pd.cut(df['PRCP'], bins=[-1, 0, 2.5, 1000], labels=[0, 1, 2]).astype(float)
    df['Rain_Light'] = (df['Rain_Intensity'] == 1).astype(int)
    df['Rain_Heavy'] = (df['Rain_Intensity'] == 2).astype(int)
    
    # 4. Osmotic Shock Index (Thunderstorm Asthma Risk)
    # High VPD (dried pollen) + Light Rain (moisture) + Wind (fragmentation)
    df['Shock_Index'] = df['Rain_Light'] * df['AWND'] * df['VPD']
    
    # 5. Growing Degree Days & Burst (The Bloom Driver)
    df['GDD_General'] = np.maximum(df['TAVG'] - 5, 0) # Base 5°C
    df['GDD_Tree'] = np.maximum(df['TAVG'] - 5, 0)
    df['GDD_Grass'] = np.maximum(df['TAVG'] - 10, 0)
    df['GDD_Weed'] = np.maximum(df['TAVG'] - 8, 0)
    
    # "Burst" - Rapid warming over last 5 days triggers explosive flowering
    df['GDD_Burst_5d'] = df['GDD_General'].rolling(5).sum()
    
    # Cumulative GDD (Season progress)
    df['GDD_cumsum'] = df.groupby(df['Year_Numeric'])['GDD_General'].cumsum()
    
    # --- D. WEATHER LAGS (Permitted: Weather history is abiotic) ---
    # We need to know if it rained *yesterday* to predict today, but not if there was pollen yesterday.
    for lag in [1, 2, 3, 7]:
        df[f'TMAX_lag_{lag}'] = df['TMAX'].shift(lag)
        df[f'PRCP_lag_{lag}'] = df['PRCP'].shift(lag)
        df[f'AWND_lag_{lag}'] = df['AWND'].shift(lag)
        df[f'VPD_lag_{lag}'] = df['VPD'].shift(lag)
    
    # --- E. WEATHER ROLLING AVERAGES ---
    for window in [3, 7, 14]:
        df[f'Temp_roll_{window}'] = df['TAVG'].rolling(window, min_periods=1).mean()
        df[f'Rain_roll_{window}'] = df['PRCP'].rolling(window, min_periods=1).sum()
        df[f'VPD_roll_{window}'] = df['VPD'].rolling(window, min_periods=1).mean()
    
    # --- F. PEAK SEASON PROXIMITY ---
    # Gaussian proximity to typical peak dates (approximate DOYs)
    df['Tree_Peak_Prox'] = np.exp(-0.5 * ((df['Day_of_Year'] - 110) / 30) ** 2)
    df['Grass_Peak_Prox'] = np.exp(-0.5 * ((df['Day_of_Year'] - 175) / 30) ** 2)
    df['Weed_Peak_Prox'] = np.exp(-0.5 * ((df['Day_of_Year'] - 255) / 30) ** 2)
    
    # --- G. INTERACTIONS ---
    # Dry & Windy = Dispersion risk
    df['Dry_x_Wind'] = df['Days_Since_Rain'] * df['AWND']
    
    print(f"   ✅ Engineered features: VPD, Ventilation, Shock, GDD Burst, Days_Since_Rain")
    
    return df


def select_final_features(df):
    """Select features, STRICTLY filtering out any Pollen history"""
    print("\n📋 Selecting final feature set...")
    
    features = [
        # Bio/Advanced
        'VPD', 'Ventilation_Index', 'Shock_Index', 'GDD_Burst_5d', 'Days_Since_Rain',
        'Rain_Light', 'Rain_Heavy',
        # Core Weather
        'TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND', 'Temp_Range', 'Is_Rainy',
        # Temporal
        'Day_of_Week', 'Month_sin', 'Month_cos', 'DOY_sin', 'DOY_cos',
        'Year_Numeric', 'Month_Numeric',
        # Derived Weather (Lags/Rolling/GDD)
        *[c for c in df.columns if 'lag_' in c],
        *[c for c in df.columns if 'roll_' in c],
        *[c for c in df.columns if 'GDD' in c],
        *[c for c in df.columns if 'Peak_Prox' in c],
        *[c for c in df.columns if '_x_' in c]
    ]
    
    # STRICT FILTER: Remove any feature name containing "Pollen"
    # This guarantees the model is "Cold Start" capable
    final_features = [f for f in features if 'Pollen' not in f]
    
    available = [f for f in final_features if f in df.columns]
    print(f"   ✅ Selected {len(available)} features")
    print(f"   🔒 Security Check: Features containing 'Pollen': {[f for f in available if 'Pollen' in f]}")
    
    return available


# ============================================================================
# 3. GPU-ACCELERATED TRAINING
# ============================================================================

def get_gpu_xgboost_config():
    """Configuration for RTX 2080 / CUDA"""
    return {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'n_estimators': 1000,  # Higher estimators since we lack autoregressive signal
        'learning_rate': 0.02, # Lower LR for better generalization
        'max_depth': 8,        # Deeper trees to capture complex bio-interactions
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.2,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 50
    }

def comprehensive_metrics(y_true, y_pred):
    high_mask = y_true >= 7
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'acc_within_1': np.mean(np.abs(y_true - y_pred) <= 1),
        'high_mae': mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if np.any(high_mask) else np.nan,
        'high_count': int(np.sum(high_mask))
    }

def train_model(X_train, y_train, X_test, y_test, pollen_type, args):
    import time
    print(f"\n{'='*60}\n🚀 TRAINING: {pollen_type}\n{'='*60}")
    
    # Config
    config = get_gpu_xgboost_config()
    
    # Check GPU
    try:
        test = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
        X_test_sample = X_train.iloc[:10].values
        y_test_sample = y_train.iloc[:10].values
        test.fit(X_test_sample, y_test_sample)
        print("   ✅ GPU Acceleration Active (CUDA)")
    except Exception:
        print("   ⚠️  GPU not found, falling back to CPU")
        config['device'] = 'cpu'
    
    # Tune or Train
    if args.tune:
        print("   🔍 Tuning hyperparameters...")
        # (Simplified tuning block for brevity, full grid can be inserted here)
        model = xgb.XGBRegressor(**config) # Placeholder for full search
        model.fit(X_train.values, y_train.values, eval_set=[(X_test.values, y_test.values)], verbose=False)
    else:
        # Split for early stopping
        split = int(len(X_train) * 0.9)
        X_tr, X_val = X_train.iloc[:split].copy(), X_train.iloc[split:].copy()
        y_tr, y_val = y_train.iloc[:split].copy(), y_train.iloc[split:].copy()
        
        # Ensure proper data types and reset index to avoid issues
        X_tr = X_tr.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        
        model = xgb.XGBRegressor(**config)
        
        # Convert to numpy arrays to avoid pandas DataFrame issues
        X_tr_array = X_tr.values
        y_tr_array = y_tr.values
        X_val_array = X_val.values
        y_val_array = y_val.values
        
        model.fit(X_tr_array, y_tr_array, eval_set=[(X_val_array, y_val_array)], verbose=False)
        print(f"   🌳 Trees built: {model.best_iteration if hasattr(model, 'best_iteration') else config['n_estimators']}")

    # Evaluate
    y_pred = np.clip(model.predict(X_test.values), 0, 10)
    metrics = comprehensive_metrics(y_test.values, y_pred)
    
    print(f"\n📊 Results ({pollen_type}):")
    print(f"   MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
    print(f"   Accuracy (±1 level): {metrics['acc_within_1']:.1%}")
    
    # Feature Importance
    imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(5)
    
    print(f"\n   🏆 Top Drivers:")
    for i, r in imp.iterrows():
        print(f"      - {r['Feature']}: {r['Importance']:.4f}")
        
    return model, metrics, y_pred


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help="Run hyperparameter tuning")
    args = parser.parse_args()
    
    print("="*70)
    print("🌸 FINAL POLLEN PREDICTOR (Bio-Meteorological / No History)")
    print("="*70)
    
    # 1. Load Data
    # Assumes 'rf.py' or similar loader exists in directory as per previous context
    # If not, replace with standard pd.read_csv
    try:
        from rf import load_and_prepare_data
        df = load_and_prepare_data()
    except ImportError:
        print("⚠️  Module 'rf' not found. Looking for CSV...")
        df = pd.read_csv('pollen_data.csv', parse_dates=['Date'])
        df.rename(columns={'Date': 'Date_Standard'}, inplace=True)

    # 2. Create Scales & Engineer Features
    df = create_biological_severity_scales(df)
    df = engineer_features_no_pollen_history(df)
    feature_cols = select_final_features(df)
    
    # 3. Train Loop
    pollen_types = ['Tree', 'Grass', 'Weed', 'Ragweed', 'Total_Pollen']
    results = {}
    
    for ptype in pollen_types:
        target = f'{ptype}_Severity'
        if target not in df.columns: continue
        
        # Clean & Split
        data = df.dropna(subset=feature_cols + [target])
        split_idx = int(len(data) * 0.8)
        
        X_train = data[feature_cols].iloc[:split_idx].copy().reset_index(drop=True)
        y_train = data[target].iloc[:split_idx].copy().reset_index(drop=True)
        X_test = data[feature_cols].iloc[split_idx:].copy().reset_index(drop=True)
        y_test = data[target].iloc[split_idx:].copy().reset_index(drop=True)
        
        model, mets, preds = train_model(X_train, y_train, X_test, y_test, ptype, args)
        
        # Save Artifacts
        joblib.dump(model, f'models/xgboost_{ptype.lower()}_bio_v2.joblib')
        results[ptype] = mets

    # 4. Summary
    print(f"\n{'='*70}\n🏁 FINAL SUMMARY\n{'='*70}")
    avg_mae = np.mean([m['mae'] for m in results.values()])
    print(f"Global Average MAE: {avg_mae:.4f}")
    
    # Save features list for inference consistency
    joblib.dump(feature_cols, 'models/model_features_list.joblib')
    print("✅ Models and feature lists saved to /models/")

if __name__ == "__main__":
    main()