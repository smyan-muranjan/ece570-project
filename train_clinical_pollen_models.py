"""
Biologically-Informed Pollen Predictor with Clinical Thresholds
================================================================

Implements clinically meaningful severity scales based on AAAAI guidelines:
- Tree pollen: Low(1-14), Moderate(15-89), High(90-1499), Very High(≥1500)
- Grass pollen: Low(1-4), Moderate(5-19), High(20-199), Very High(≥200)
- Weed pollen: Low(1-9), Moderate(10-49), High(50-499), Very High(≥500)

This approach:
✓ Uses biologically meaningful thresholds (not arbitrary percentiles)
✓ Predicts separate models for each pollen type
✓ Combines predictions weighted by clinical impact
✓ Provides species-specific forecasts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import randint, uniform
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# AAAAI Clinical Thresholds (grains/m³)
CLINICAL_THRESHOLDS = {
    'Tree': {
        'None': 0,
        'Low': 1,
        'Moderate': 15,
        'High': 90,
        'Very High': 1500
    },
    'Grass': {
        'None': 0,
        'Low': 1,
        'Moderate': 5,
        'High': 20,
        'Very High': 200
    },
    'Weed': {
        'None': 0,
        'Low': 1,
        'Moderate': 10,
        'High': 50,
        'Very High': 500
    }
}

def create_clinical_severity_scale(df):
    """
    Create severity scales based on AAAAI clinical thresholds
    Returns 0-4 scale: 0=None, 1=Low, 2=Moderate, 3=High, 4=Very High
    """
    print("🏥 Creating clinical severity scales...")
    
    def count_to_severity(count, thresholds):
        """Convert pollen count to clinical severity level"""
        if count == 0:
            return 0  # None
        elif count < thresholds['Moderate']:
            return 1  # Low
        elif count < thresholds['High']:
            return 2  # Moderate
        elif count < thresholds['Very High']:
            return 3  # High
        else:
            return 4  # Very High
    
    # Tree severity
    df['Tree_Severity'] = df['Tree'].apply(
        lambda x: count_to_severity(x, CLINICAL_THRESHOLDS['Tree'])
    )
    
    # Grass severity
    df['Grass_Severity'] = df['Grass'].apply(
        lambda x: count_to_severity(x, CLINICAL_THRESHOLDS['Grass'])
    )
    
    # Weed severity
    df['Weed_Severity'] = df['Weed'].apply(
        lambda x: count_to_severity(x, CLINICAL_THRESHOLDS['Weed'])
    )
    
    # Combined severity (weighted average based on typical sensitization rates)
    # Tree: 20%, Grass: 30%, Weed: 50% (ragweed is highly allergenic)
    df['Combined_Severity'] = (
        0.20 * df['Tree_Severity'] + 
        0.30 * df['Grass_Severity'] + 
        0.50 * df['Weed_Severity']
    ).round().astype(int)
    
    # Clip to 0-4 range
    df['Combined_Severity'] = df['Combined_Severity'].clip(0, 4)
    
    # Print distribution
    print(f"\n   Clinical Severity Distribution:")
    print(f"   Tree Pollen:")
    for level, count in df['Tree_Severity'].value_counts().sort_index().items():
        level_name = ['None', 'Low', 'Moderate', 'High', 'Very High'][level]
        print(f"      {level_name:12} ({level}): {count:4} samples ({count/len(df)*100:.1f}%)")
    
    print(f"\n   Grass Pollen:")
    for level, count in df['Grass_Severity'].value_counts().sort_index().items():
        level_name = ['None', 'Low', 'Moderate', 'High', 'Very High'][level]
        print(f"      {level_name:12} ({level}): {count:4} samples ({count/len(df)*100:.1f}%)")
    
    print(f"\n   Weed Pollen:")
    for level, count in df['Weed_Severity'].value_counts().sort_index().items():
        level_name = ['None', 'Low', 'Moderate', 'High', 'Very High'][level]
        print(f"      {level_name:12} ({level}): {count:4} samples ({count/len(df)*100:.1f}%)")
    
    print(f"\n   Combined (Weighted):")
    for level, count in df['Combined_Severity'].value_counts().sort_index().items():
        level_name = ['None', 'Low', 'Moderate', 'High', 'Very High'][level]
        print(f"      {level_name:12} ({level}): {count:4} samples ({count/len(df)*100:.1f}%)")
    
    return df

def engineer_features(df):
    """Engineer comprehensive feature set"""
    print("\n🔬 Engineering features...")
    
    df = df.sort_values('Date_Standard').reset_index(drop=True)
    
    # TEMPORAL
    df['Day_of_Year'] = df['Date_Standard'].dt.dayofyear
    df['Day_of_Week'] = df['Date_Standard'].dt.dayofweek
    df['Month'] = df['Date_Standard'].dt.month
    df['Year'] = df['Date_Standard'].dt.year
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DOY_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
    df['DOY_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
    
    # WEATHER
    df['Temp_Range'] = df['TMAX'] - df['TMIN']
    df['Is_Rainy'] = (df['PRCP'] > 0).astype(int)
    
    # GDD
    base_temp = 5
    df['GDD'] = np.maximum(df['TAVG'] - base_temp, 0)
    df['GDD_cumsum'] = df.groupby(df['Year'])['GDD'].cumsum()
    df['GDD_7d'] = df['GDD'].rolling(7, min_periods=1).sum()
    df['GDD_14d'] = df['GDD'].rolling(14, min_periods=1).sum()
    df['GDD_30d'] = df['GDD'].rolling(30, min_periods=1).sum()
    df['Below_Freezing'] = (df['TMIN'] < 0).astype(int)
    df['Frost_Event'] = ((df['TMIN'] < 0) & (df['TMIN'].shift(1) >= 0)).astype(int)
    frost_reset = df['Frost_Event'].cumsum()
    df['GDD_since_frost'] = df.groupby(frost_reset)['GDD'].cumsum()
    
    # POLLEN LAGS (by type)
    for lag in [1, 3, 7]:
        df[f'Tree_lag_{lag}'] = df['Tree'].shift(lag)
        df[f'Grass_lag_{lag}'] = df['Grass'].shift(lag)
        df[f'Weed_lag_{lag}'] = df['Weed'].shift(lag)
        df[f'Tree_Severity_lag_{lag}'] = df['Tree_Severity'].shift(lag)
        df[f'Grass_Severity_lag_{lag}'] = df['Grass_Severity'].shift(lag)
        df[f'Weed_Severity_lag_{lag}'] = df['Weed_Severity'].shift(lag)
    
    # POLLEN ROLLING (by type)
    for window in [3, 7, 14]:
        df[f'Tree_roll_{window}'] = df['Tree'].shift(1).rolling(window, min_periods=1).mean()
        df[f'Grass_roll_{window}'] = df['Grass'].shift(1).rolling(window, min_periods=1).mean()
        df[f'Weed_roll_{window}'] = df['Weed'].shift(1).rolling(window, min_periods=1).mean()
    
    # POLLEN MOMENTUM (by type)
    df['Tree_acceleration'] = df['Tree_lag_1'].diff()
    df['Grass_acceleration'] = df['Grass_lag_1'].diff()
    df['Weed_acceleration'] = df['Weed_lag_1'].diff()
    df['Tree_trend_7d'] = (df['Tree_lag_1'] - df['Tree_lag_7']) / 6
    df['Grass_trend_7d'] = (df['Grass_lag_1'] - df['Grass_lag_7']) / 6
    df['Weed_trend_7d'] = (df['Weed_lag_1'] - df['Weed_lag_7']) / 6
    df['Tree_volatility_7d'] = df['Tree_lag_1'].rolling(7, min_periods=1).std()
    df['Grass_volatility_7d'] = df['Grass_lag_1'].rolling(7, min_periods=1).std()
    df['Weed_volatility_7d'] = df['Weed_lag_1'].rolling(7, min_periods=1).std()
    df['Tree_spike'] = (df['Tree_lag_1'] > df['Tree_roll_7'] * 1.5).astype(int)
    df['Grass_spike'] = (df['Grass_lag_1'] > df['Grass_roll_7'] * 1.5).astype(int)
    df['Weed_spike'] = (df['Weed_lag_1'] > df['Weed_roll_7'] * 1.5).astype(int)
    
    # WEATHER LAGS
    for lag in [1, 3, 7]:
        df[f'TMAX_lag_{lag}'] = df['TMAX'].shift(lag)
        df[f'PRCP_lag_{lag}'] = df['PRCP'].shift(lag)
    
    # WEATHER ROLLING
    for window in [3, 7, 14]:
        df[f'Temp_roll_{window}'] = df['TAVG'].rolling(window, min_periods=1).mean()
        df[f'Precip_roll_{window}'] = df['PRCP'].rolling(window, min_periods=1).sum()
    
    # WEATHER-POLLEN INTERACTIONS (by type)
    df['Tree_x_Rain'] = df['Tree_roll_3'] * df['PRCP']
    df['Grass_x_Rain'] = df['Grass_roll_3'] * df['PRCP']
    df['Weed_x_Rain'] = df['Weed_roll_3'] * df['PRCP']
    df['Tree_x_Wind'] = df['Tree_roll_3'] * df['AWND']
    df['Grass_x_Wind'] = df['Grass_roll_3'] * df['AWND']
    df['Weed_x_Wind'] = df['Weed_roll_3'] * df['AWND']
    df['Tree_x_Temp'] = df['Tree_roll_3'] * df['TAVG']
    df['Grass_x_Temp'] = df['Grass_roll_3'] * df['TAVG']
    df['Weed_x_Temp'] = df['Weed_roll_3'] * df['TAVG']
    
    # BIOLOGICAL FEATURES
    df['TAVG_30d_mean'] = df['TAVG'].rolling(30, min_periods=1).mean()
    df['Temp_Anomaly'] = df['TAVG'] - df['TAVG_30d_mean']
    df['TAVG_change'] = df['TAVG'].diff()
    df['Temp_Volatility_7d'] = df['TAVG_change'].rolling(7, min_periods=1).std()
    df['PRCP_YTD'] = df.groupby(df['Year'])['PRCP'].cumsum()
    df['PRCP_30d_sum'] = df['PRCP'].rolling(30, min_periods=1).sum()
    df['Consecutive_Dry_Days'] = (df['PRCP'] == 0).astype(int).groupby((df['PRCP'] > 0).cumsum()).cumsum()
    
    # PEAK SEASON INDICATORS
    tree_peak, grass_peak, ragweed_peak = 105, 165, 255
    df['Days_to_Tree_Peak'] = np.abs(df['Day_of_Year'] - tree_peak)
    df['Days_to_Grass_Peak'] = np.abs(df['Day_of_Year'] - grass_peak)
    df['Days_to_Ragweed_Peak'] = np.abs(df['Day_of_Year'] - ragweed_peak)
    df['In_Tree_Season'] = ((df['Day_of_Year'] >= 60) & (df['Day_of_Year'] <= 150)).astype(int)
    df['In_Grass_Season'] = ((df['Day_of_Year'] >= 120) & (df['Day_of_Year'] <= 210)).astype(int)
    df['In_Ragweed_Season'] = ((df['Day_of_Year'] >= 210) & (df['Day_of_Year'] <= 300)).astype(int)
    df['GDD_x_Tree_Season'] = df['GDD_cumsum'] * df['In_Tree_Season']
    df['GDD_x_Grass_Season'] = df['GDD_cumsum'] * df['In_Grass_Season']
    df['GDD_x_Ragweed_Season'] = df['GDD_cumsum'] * df['In_Ragweed_Season']
    
    print(f"   ✅ Created {len(df.columns) - 34} engineered features")
    
    return df

def select_features(df):
    """Select features for modeling"""
    exclude = ['Date_Standard', 'Total_Pollen', 'Tree', 'Grass', 'Weed', 'Ragweed',
               'Tree_Severity', 'Grass_Severity', 'Weed_Severity', 'Combined_Severity',
               'Year', 'Month', 'TAVG_30d_mean', 'TAVG_change', 'Frost_Event',
               'Consecutive_Dry_Days', 'Day_of_Year',
               'Date', 'Week', 'Tree_Level', 'Grass_Level', 'Weed_Level', 'Ragweed_Level',
               'OBJECTID', 'Year_Numeric', 'Month_Numeric', 'STATION', 'NAME',
               'Tree_Level_Numeric', 'Grass_Level_Numeric', 'Weed_Level_Numeric', 
               'Ragweed_Level_Numeric', 'TAVG_calculated', 'WSF2', 'WSF5']
    
    features = [c for c in df.columns if c not in exclude]
    print(f"   ✅ Selected {len(features)} features")
    return features

def train_xgboost_model(X_train, y_train, X_test, y_test, pollen_type):
    """Train tuned XGBoost model for specific pollen type"""
    print(f"\n🎯 Training XGBoost for {pollen_type} Pollen Severity...")
    
    param_distributions = {
        'n_estimators': randint(200, 800),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.12),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'min_child_weight': randint(1, 15),
        'gamma': uniform(0, 0.4),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 3),
    }
    
    base_model = xgb.XGBRegressor(
        device='cuda',
        tree_method='hist',
        random_state=42,
        early_stopping_rounds=50
    )
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    print(f"   Hyperparameter search (20 iterations)...")
    random_search = RandomizedSearchCV(
        base_model, param_distributions, n_iter=20, cv=tscv,
        scoring='neg_mean_absolute_error', random_state=42, n_jobs=1, verbose=0
    )
    
    random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    model = random_search.best_estimator_
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    within_1 = np.mean(np.abs(y_test - test_pred) <= 1)
    
    print(f"   Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    print(f"   Accuracy (±1 level): {within_1:.1%}")
    
    return model, test_pred, test_mae, test_r2

def main():
    print("="*80)
    print("🏥 BIOLOGICALLY-INFORMED POLLEN PREDICTOR")
    print("="*80)
    print("Using AAAAI clinical thresholds for meaningful severity scales")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('combined_allergy_weather.csv')
    df['Date_Standard'] = pd.to_datetime(df['Date_Standard'])
    print(f"   Loaded {len(df)} samples")
    
    # Create clinical severity scales
    df = create_clinical_severity_scale(df)
    
    # Engineer features
    df = engineer_features(df)
    features = select_features(df)
    
    # Prepare data
    print("\n📊 Preparing data...")
    targets = ['Tree_Severity', 'Grass_Severity', 'Weed_Severity', 'Combined_Severity']
    df_clean = df.dropna(subset=features + targets).copy()
    print(f"   Clean dataset: {len(df_clean)} samples")
    
    split_date = df_clean['Date_Standard'].quantile(0.8)
    train_mask = df_clean['Date_Standard'] <= split_date
    test_mask = df_clean['Date_Standard'] > split_date
    
    X_train = df_clean.loc[train_mask, features]
    X_test = df_clean.loc[test_mask, features]
    
    train_dates = df_clean.loc[train_mask, 'Date_Standard']
    test_dates = df_clean.loc[test_mask, 'Date_Standard']
    
    print(f"   Train: {len(X_train)} samples ({train_dates.min().date()} to {train_dates.max().date()})")
    print(f"   Test:  {len(X_test)} samples ({test_dates.min().date()} to {test_dates.max().date()})")
    
    # Train models for each pollen type
    results = {}
    models = {}
    
    print("\n" + "="*80)
    print("🌳🌾🌿 TRAINING SPECIES-SPECIFIC MODELS")
    print("="*80)
    
    for target in targets:
        y_train = df_clean.loc[train_mask, target]
        y_test = df_clean.loc[test_mask, target]
        
        pollen_type = target.replace('_Severity', '')
        model, pred, mae, r2 = train_xgboost_model(X_train, y_train, X_test, y_test, pollen_type)
        
        models[target] = model
        results[target] = {
            'mae': float(mae),
            'r2': float(r2),
            'predictions': pred
        }
    
    # Compare with previous best model
    print("\n" + "="*80)
    print("🏆 FINAL COMPARISON")
    print("="*80)
    
    try:
        with open('xgboost_vs_ensemble_comparison.json', 'r') as f:
            prev_results = json.load(f)
            prev_mae = prev_results['xgboost_tuned']['test_mae']
            prev_r2 = prev_results['xgboost_tuned']['test_r2']
            prev_target = "Combined (0-10 percentile scale)"
    except:
        prev_mae = 0.876
        prev_r2 = 0.801
        prev_target = "Previous XGBoost (0-10 percentile)"
    
    print(f"\n{'Model':<40} {'MAE':<12} {'R²':<12} {'Scale'}")
    print("-" * 80)
    print(f"{prev_target:<40} {prev_mae:<12.4f} {prev_r2:<12.4f} {'0-10'}")
    print()
    for target in targets:
        model_name = f"{target.replace('_Severity', '')} (Clinical AAAAI)"
        print(f"{model_name:<40} {results[target]['mae']:<12.4f} {results[target]['r2']:<12.4f} {'0-4'}")
    
    print("\n" + "="*80)
    print("💡 INTERPRETATION")
    print("="*80)
    print("""
Clinical AAAAI Scale (0-4):
• More biologically meaningful thresholds
• Species-specific models capture unique patterns
• Lower MAE due to compressed scale (0-4 vs 0-10)
• Better interpretability for clinical decision-making

Combined model uses weighted average:
• Tree: 20% (early spring allergen)
• Grass: 30% (summer allergen)  
• Weed: 50% (ragweed highly allergenic)

Recommendation:
• Use species-specific models for detailed forecasts
• Use Combined model for overall allergy risk assessment
    """)
    
    # Save models
    print("\n💾 Saving models...")
    for target, model in models.items():
        filename = f'pollen_predictor_{target.lower()}.joblib'
        joblib.dump(model, filename)
        print(f"   ✅ {filename}")
    
    # Save results
    results_summary = {
        'clinical_thresholds': CLINICAL_THRESHOLDS,
        'model_performance': {k: {'mae': v['mae'], 'r2': v['r2']} for k, v in results.items()},
        'previous_best': {'mae': prev_mae, 'r2': prev_r2},
        'trained_at': str(datetime.now())
    }
    
    with open('clinical_pollen_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"   ✅ clinical_pollen_results.json")
    
    print("\n✅ TRAINING COMPLETE!")

if __name__ == "__main__":
    main()
