"""
Weather-Only Pollen Predictor
Forces the model to predict pollen from weather features only (no pollen lags)
This represents a true forecasting scenario where future pollen must be predicted
from weather conditions alone.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path='combined_allergy_weather.csv'):
    """Load data and prepare features for pollen prediction"""
    print("📂 Loading and preparing data...")
    
    df = pd.read_csv(file_path)
    df['Date_Standard'] = pd.to_datetime(df['Date_Standard'])
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Date range: {df['Date_Standard'].min().date()} to {df['Date_Standard'].max().date()}")
    
    return df

def create_pollen_severity_scale(df):
    """Create 0-10 pollen severity scale based on percentiles"""
    print("\n🌸 Creating pollen severity scale (0-10)...")
    
    pollen_data = df['Total_Pollen'].copy()
    
    percentiles = [0, 10, 25, 40, 55, 70, 80, 85, 90, 95, 100]
    thresholds = np.percentile(pollen_data[pollen_data > 0], percentiles)
    
    print("   Pollen Severity Scale:")
    severity_labels = ['None', 'Very Low', 'Low', 'Low-Mod', 'Moderate', 
                      'Mod-High', 'High', 'Very High', 'Extreme', 'Severe']
    
    for i, (label, threshold) in enumerate(zip(severity_labels, thresholds)):
        print(f"   {i}: {label:10} >= {threshold:6.1f} pollen count")
    
    def pollen_to_severity(pollen_count):
        if pollen_count == 0:
            return 0
        for i in range(len(thresholds)-1, 0, -1):
            if pollen_count >= thresholds[i]:
                return i
        return 0
    
    df['Pollen_Severity'] = df['Total_Pollen'].apply(pollen_to_severity)
    
    return df, thresholds

def engineer_weather_features(df):
    """
    Engineer features from WEATHER ONLY (no pollen lags)
    
    This creates a true forecasting scenario where we must predict pollen
    from atmospheric conditions, seasonal patterns, and plant phenology.
    """
    print("\n🌤️  Engineering WEATHER-ONLY features...")
    
    df = df.sort_values('Date_Standard').reset_index(drop=True)
    
    # ========================================================================
    # Temporal Features (seasonal patterns)
    # ========================================================================
    df['Year'] = df['Date_Standard'].dt.year
    df['Day_of_Year'] = df['Date_Standard'].dt.dayofyear
    df['Day_of_Week'] = df['Date_Standard'].dt.dayofweek
    df['Week_of_Year'] = df['Date_Standard'].dt.isocalendar().week
    
    # Cyclical encoding for perfect seasonality
    df['Month_sin'] = np.sin(2 * np.pi * df['Date_Standard'].dt.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Date_Standard'].dt.month / 12)
    df['DOY_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
    df['DOY_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
    
    print("   ✅ Temporal features (cyclical encoding)")
    
    # ========================================================================
    # Core Weather Features
    # ========================================================================
    df['Temp_Range'] = df['TMAX'] - df['TMIN']
    df['Is_Rainy'] = (df['PRCP'] > 0).astype(int)
    df['Is_Heavy_Rain'] = (df['PRCP'] > df['PRCP'].quantile(0.75)).astype(int)
    
    # ========================================================================
    # Temperature Dynamics (key for plant development)
    # ========================================================================
    # 30-day rolling mean (seasonal norm)
    df['TAVG_30d_mean'] = df['TAVG'].rolling(window=30, min_periods=1).mean()
    df['Temp_Anomaly'] = df['TAVG'] - df['TAVG_30d_mean']
    
    # Temperature change (weather fronts)
    df['TAVG_change'] = df['TAVG'].diff()
    df['Temp_Volatility_7d'] = df['TAVG_change'].rolling(window=7, min_periods=1).std()
    
    # Extreme temperature indicators
    df['Above_20C'] = (df['TAVG'] > 20).astype(int)
    df['Below_5C'] = (df['TAVG'] < 5).astype(int)
    df['Freezing'] = (df['TMIN'] < 0).astype(int)
    
    print("   ✅ Temperature dynamics (anomalies, volatility, extremes)")
    
    # ========================================================================
    # Precipitation Patterns (soil moisture, pollen suppression)
    # ========================================================================
    # Cumulative rainfall (soil moisture proxy)
    df['PRCP_YTD'] = df.groupby(df['Date_Standard'].dt.year)['PRCP'].cumsum()
    df['PRCP_30d_sum'] = df['PRCP'].rolling(window=30, min_periods=1).sum()
    df['PRCP_7d_sum'] = df['PRCP'].rolling(window=7, min_periods=1).sum()
    
    # Dry/wet spell indicators
    df['Days_Since_Rain'] = (df['PRCP'] == 0).astype(int).groupby((df['PRCP'] > 0).cumsum()).cumsum()
    df['Consecutive_Dry_Days'] = df['Days_Since_Rain'].rolling(window=7, min_periods=1).max()
    
    print("   ✅ Precipitation patterns (soil moisture, dry spells)")
    
    # ========================================================================
    # Wind Patterns (pollen dispersal)
    # ========================================================================
    df['AWND_7d_mean'] = df['AWND'].rolling(window=7, min_periods=1).mean()
    df['AWND_30d_pct'] = df['AWND'].rolling(window=30, min_periods=1).apply(
        lambda x: (x.iloc[-1] / x.max() * 100) if len(x) > 0 and x.max() > 0 else 0
    )
    df['High_Wind'] = (df['AWND'] > df['AWND'].quantile(0.75)).astype(int)
    
    print("   ✅ Wind patterns (dispersal potential)")
    
    # ========================================================================
    # Growing Degree Days (plant phenology - CRITICAL for pollen)
    # ========================================================================
    base_temp = 5  # Base temperature for plant growth
    df['GDD'] = np.maximum(df['TAVG'] - base_temp, 0)
    
    # Annual cumulative GDD (total heat accumulation)
    df['GDD_cumsum'] = df.groupby(df['Date_Standard'].dt.year)['GDD'].cumsum()
    
    # Recent GDD (immediate growth activity)
    df['GDD_7d'] = df['GDD'].rolling(window=7, min_periods=1).sum()
    df['GDD_14d'] = df['GDD'].rolling(window=14, min_periods=1).sum()
    df['GDD_30d'] = df['GDD'].rolling(window=30, min_periods=1).sum()
    
    # GDD with frost reset (biologically accurate)
    df['Frost_Event'] = ((df['TMIN'] < 0) & (df['TMIN'].shift(1) >= 0)).astype(int)
    frost_reset = df['Frost_Event'].cumsum()
    df['GDD_since_frost'] = df.groupby(frost_reset)['GDD'].cumsum()
    
    # GDD thresholds (known phenological triggers)
    df['GDD_cumsum_above_100'] = (df['GDD_cumsum'] > 100).astype(int)
    df['GDD_cumsum_above_200'] = (df['GDD_cumsum'] > 200).astype(int)
    df['GDD_cumsum_above_500'] = (df['GDD_cumsum'] > 500).astype(int)
    
    print("   ✅ Growing Degree Days (plant phenology engine)")
    
    # ========================================================================
    # Peak Season Proximity (biological timing)
    # ========================================================================
    # Tree pollen: March-May (DOY 60-150, peak ~105)
    tree_peak = 105
    df['Days_to_Tree_Peak'] = np.abs(df['Day_of_Year'] - tree_peak)
    df['In_Tree_Season'] = ((df['Day_of_Year'] >= 60) & (df['Day_of_Year'] <= 150)).astype(int)
    df['Tree_Season_Progress'] = np.clip((df['Day_of_Year'] - 60) / 90, 0, 1)
    
    # Grass pollen: May-July (DOY 120-210, peak ~165)
    grass_peak = 165
    df['Days_to_Grass_Peak'] = np.abs(df['Day_of_Year'] - grass_peak)
    df['In_Grass_Season'] = ((df['Day_of_Year'] >= 120) & (df['Day_of_Year'] <= 210)).astype(int)
    df['Grass_Season_Progress'] = np.clip((df['Day_of_Year'] - 120) / 90, 0, 1)
    
    # Ragweed: August-October (DOY 210-300, peak ~255)
    ragweed_peak = 255
    df['Days_to_Ragweed_Peak'] = np.abs(df['Day_of_Year'] - ragweed_peak)
    df['In_Ragweed_Season'] = ((df['Day_of_Year'] >= 210) & (df['Day_of_Year'] <= 300)).astype(int)
    df['Ragweed_Season_Progress'] = np.clip((df['Day_of_Year'] - 210) / 90, 0, 1)
    
    print("   ✅ Peak season indicators (tree, grass, ragweed)")
    
    # ========================================================================
    # Weather Lags (recent weather influences current pollen release)
    # ========================================================================
    for lag in [1, 3, 7]:
        df[f'TMAX_lag_{lag}'] = df['TMAX'].shift(lag)
        df[f'TMIN_lag_{lag}'] = df['TMIN'].shift(lag)
        df[f'PRCP_lag_{lag}'] = df['PRCP'].shift(lag)
        df[f'AWND_lag_{lag}'] = df['AWND'].shift(lag)
        df[f'GDD_lag_{lag}'] = df['GDD'].shift(lag)
    
    print("   ✅ Weather lags (1, 3, 7 days)")
    
    # ========================================================================
    # Rolling Weather Averages (recent conditions)
    # ========================================================================
    for window in [3, 7, 14]:
        df[f'TAVG_roll_{window}'] = df['TAVG'].rolling(window=window, min_periods=1).mean()
        df[f'PRCP_roll_{window}'] = df['PRCP'].rolling(window=window, min_periods=1).sum()
        df[f'AWND_roll_{window}'] = df['AWND'].rolling(window=window, min_periods=1).mean()
    
    print("   ✅ Rolling weather averages (3, 7, 14 days)")
    
    # ========================================================================
    # Interaction Features (weather + phenology)
    # ========================================================================
    df['GDD_x_Tree_Season'] = df['GDD_cumsum'] * df['In_Tree_Season']
    df['GDD_x_Grass_Season'] = df['GDD_cumsum'] * df['In_Grass_Season']
    df['GDD_x_Ragweed_Season'] = df['GDD_cumsum'] * df['In_Ragweed_Season']
    
    df['Temp_x_Tree_Season'] = df['TAVG'] * df['In_Tree_Season']
    df['Temp_x_Grass_Season'] = df['TAVG'] * df['In_Grass_Season']
    
    print("   ✅ Interaction features (weather × phenology)")
    
    feature_count = len([col for col in df.columns if col not in ['Date_Standard', 'Total_Pollen', 
                                                                    'Tree_Pollen', 'Grass_Pollen', 
                                                                    'Weed_Pollen', 'Ragweed_Pollen',
                                                                    'Pollen_Severity', 'TMAX', 'TMIN', 
                                                                    'TAVG', 'PRCP', 'AWND', 'Year',
                                                                    'Day_of_Year', 'TAVG_30d_mean',
                                                                    'TAVG_change', 'Days_Since_Rain',
                                                                    'Frost_Event', 'Week_of_Year']])
    print(f"\n   ✅ Created {feature_count} weather-based features")
    print(f"   ⚠️  NO POLLEN LAGS - Pure weather forecasting!")
    
    return df

def select_weather_features(df):
    """Select comprehensive weather feature set"""
    print("\n📋 Selecting weather features...")
    
    features = [
        # Core weather
        'TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND', 'Temp_Range', 'Is_Rainy', 'Is_Heavy_Rain',
        
        # Temporal (cyclical)
        'Day_of_Week', 'Month_sin', 'Month_cos', 'DOY_sin', 'DOY_cos',
        
        # Temperature dynamics
        'Temp_Anomaly', 'TAVG_change', 'Temp_Volatility_7d',
        'Above_20C', 'Below_5C', 'Freezing',
        
        # Precipitation patterns
        'PRCP_YTD', 'PRCP_30d_sum', 'PRCP_7d_sum', 'Consecutive_Dry_Days',
        
        # Wind
        'AWND_7d_mean', 'AWND_30d_pct', 'High_Wind',
        
        # GDD (phenology engine)
        'GDD', 'GDD_cumsum', 'GDD_7d', 'GDD_14d', 'GDD_30d', 'GDD_since_frost',
        'GDD_cumsum_above_100', 'GDD_cumsum_above_200', 'GDD_cumsum_above_500',
        
        # Peak season indicators
        'Days_to_Tree_Peak', 'In_Tree_Season', 'Tree_Season_Progress',
        'Days_to_Grass_Peak', 'In_Grass_Season', 'Grass_Season_Progress',
        'Days_to_Ragweed_Peak', 'In_Ragweed_Season', 'Ragweed_Season_Progress',
        
        # Weather lags
        *[c for c in df.columns if '_lag_' in c and 'Pollen' not in c],
        
        # Rolling weather
        *[c for c in df.columns if ('_roll_' in c or 'roll_' in c) and 'Pollen' not in c],
        
        # Interactions
        'GDD_x_Tree_Season', 'GDD_x_Grass_Season', 'GDD_x_Ragweed_Season',
        'Temp_x_Tree_Season', 'Temp_x_Grass_Season',
    ]
    
    available = [f for f in features if f in df.columns]
    
    print(f"   Selected {len(available)} features:")
    print(f"   • Core weather: 8")
    print(f"   • Temporal: 5")
    print(f"   • Temperature dynamics: 6")
    print(f"   • Precipitation: 4")
    print(f"   • Wind: 3")
    print(f"   • GDD/Phenology: {len([f for f in available if 'GDD' in f])}")
    print(f"   • Peak seasons: 9")
    print(f"   • Weather lags: {len([f for f in available if '_lag_' in f])}")
    print(f"   • Rolling weather: {len([f for f in available if 'roll_' in f or '_roll_' in f])}")
    print(f"   • Interactions: 5")
    print(f"\n   ⚠️  NO POLLEN FEATURES - Weather forecasting only!")
    
    return available

def prepare_train_test_split(df, feature_cols, target_col='Pollen_Severity'):
    """Prepare train/test split"""
    print(f"\n📊 Preparing train/test split...")
    
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    
    print(f"   Clean dataset: {len(df_clean)} samples")
    print(f"   Date range: {df_clean['Date_Standard'].min().date()} to {df_clean['Date_Standard'].max().date()}")
    
    split_date = df_clean['Date_Standard'].quantile(0.8)
    
    train_mask = df_clean['Date_Standard'] <= split_date
    test_mask = df_clean['Date_Standard'] > split_date
    
    X_train = df_clean.loc[train_mask, feature_cols]
    X_test = df_clean.loc[test_mask, feature_cols]
    y_train = df_clean.loc[train_mask, target_col]
    y_test = df_clean.loc[test_mask, target_col]
    
    train_dates = df_clean.loc[train_mask, 'Date_Standard']
    test_dates = df_clean.loc[test_mask, 'Date_Standard']
    
    print(f"   Train: {len(X_train)} samples ({train_dates.min().date()} to {train_dates.max().date()})")
    print(f"   Test:  {len(X_test)} samples ({test_dates.min().date()} to {test_dates.max().date()})")
    
    return X_train, X_test, y_train, y_test, train_dates, test_dates, df_clean

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with cross-validation"""
    print("\n🤖 Training Weather-Only Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=200,  # More trees for complex patterns
        max_depth=20,      # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                scoring='neg_mean_absolute_error', n_jobs=-1)
    
    print(f"   Cross-validation MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Accuracy within severity levels
    within_1 = np.mean(np.abs(y_test - y_test_pred) <= 1)
    within_2 = np.mean(np.abs(y_test - y_test_pred) <= 2)
    
    results = {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'cv_mae_mean': -cv_scores.mean(),
        'cv_mae_std': cv_scores.std(),
        'acc_within_1': within_1,
        'acc_within_2': within_2
    }
    
    print(f"\n   📊 Training Results:")
    print(f"      MAE:  {train_mae:.4f}")
    print(f"      R²:   {train_r2:.4f}")
    print(f"\n   📊 Test Results:")
    print(f"      MAE:  {test_mae:.4f}")
    print(f"      RMSE: {test_rmse:.4f}")
    print(f"      R²:   {test_r2:.4f}")
    print(f"      Acc (±1 level): {within_1:.1%}")
    print(f"      Acc (±2 levels): {within_2:.1%}")
    
    return model, results, y_test_pred, y_test

def compare_with_baseline():
    """Compare with baseline model"""
    print("\n" + "="*70)
    print("📊 COMPARISON WITH BASELINE")
    print("="*70)
    
    try:
        with open('model_results.json', 'r') as f:
            baseline = json.load(f)
            baseline_mae = baseline['test_mae']
            baseline_r2 = baseline['test_r2']
            print(f"\n   Baseline (with pollen lags):")
            print(f"   • MAE: {baseline_mae:.4f}")
            print(f"   • R²:  {baseline_r2:.4f}")
            return baseline_mae, baseline_r2
    except:
        print("\n   ⚠️  Baseline results not found")
        return None, None

def main():
    print("="*70)
    print("🌤️  WEATHER-ONLY POLLEN PREDICTOR")
    print("="*70)
    print("🎯 Challenge: Predict pollen from weather features ONLY")
    print("   • NO pollen lag features")
    print("   • NO pollen rolling averages")
    print("   • Pure atmospheric + phenological forecasting")
    print("="*70)
    
    # Load data
    df = load_and_prepare_data()
    df, thresholds = create_pollen_severity_scale(df)
    df = engineer_weather_features(df)
    features = select_weather_features(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test, train_dates, test_dates, df_clean = \
        prepare_train_test_split(df, features)
    
    # Train model
    model, results, y_pred, y_true = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Compare with baseline
    baseline_mae, baseline_r2 = compare_with_baseline()
    
    if baseline_mae:
        diff_mae = results['test_mae'] - baseline_mae
        diff_r2 = results['test_r2'] - baseline_r2
        pct_mae = (diff_mae / baseline_mae) * 100
        pct_r2 = (diff_r2 / baseline_r2) * 100
        
        print(f"\n   Weather-only model:")
        print(f"   • MAE: {results['test_mae']:.4f}")
        print(f"   • R²:  {results['test_r2']:.4f}")
        
        print(f"\n   📊 Difference (weather-only vs baseline):")
        print(f"   • MAE: +{diff_mae:.4f} ({pct_mae:+.1f}%)")
        print(f"   • R²:  {diff_r2:+.4f} ({pct_r2:+.1f}%)")
        
        print(f"\n   💡 Interpretation:")
        if pct_mae < 50:
            print(f"   ✅ Weather features capture most of the signal!")
        elif pct_mae < 100:
            print(f"   ⚠️  Weather helps but pollen autocorrelation is important")
        else:
            print(f"   ❌ Pollen strongly autocorrelated - weather secondary")
    
    # Save results
    print("\n💾 Saving outputs...")
    joblib.dump(model, 'pollen_predictor_weather_only.joblib')
    joblib.dump(features, 'weather_only_features.joblib')
    
    with open('weather_only_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"   ✅ Model: pollen_predictor_weather_only.joblib")
    print(f"   ✅ Results: weather_only_results.json")
    print(f"   ✅ Features: weather_only_features.joblib")
    
    print("\n🏆 Top 20 Most Important Features:")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"   {row['feature']:30} {row['importance']:.4f}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
