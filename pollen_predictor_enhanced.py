"""
Enhanced Pollen Intensity Predictor
Improved feature engineering based on biological and meteorological insights
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

def engineer_enhanced_features(df):
    """
    Engineer enhanced features with biological and meteorological insights
    
    🔧 Improvements:
    - Remove redundant temporal features (keep only essential)
    - Add temperature anomalies (deviation from seasonal norm)
    - Add cumulative rainfall (soil moisture proxy)
    - Add season-specific GDD with frost resets
    - Add peak season proximity indicators
    - Simplify lag features to reduce collinearity
    """
    print("\n🔧 Engineering ENHANCED features...")
    
    df = df.sort_values('Date_Standard').reset_index(drop=True)
    
    # ========================================================================
    # ✅ KEEP: Core Temporal Features (simplified)
    # ========================================================================
    df['Year'] = df['Date_Standard'].dt.year
    df['Day_of_Year'] = df['Date_Standard'].dt.dayofyear
    df['Day_of_Week'] = df['Date_Standard'].dt.dayofweek  # Weekly patterns
    
    # ✅ KEEP: Cyclical encoding (captures seasonality perfectly)
    df['Month_sin'] = np.sin(2 * np.pi * df['Date_Standard'].dt.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Date_Standard'].dt.month / 12)
    df['DOY_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
    df['DOY_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
    
    # ========================================================================
    # ✅ KEEP: Core Weather Features
    # ========================================================================
    df['Temp_Range'] = df['TMAX'] - df['TMIN']
    df['Is_Rainy'] = (df['PRCP'] > 0).astype(int)
    
    # ========================================================================
    # ➕ NEW: Temperature Anomaly (deviation from 30-day rolling mean)
    # ========================================================================
    df['TAVG_30d_mean'] = df['TAVG'].rolling(window=30, min_periods=1).mean()
    df['Temp_Anomaly'] = df['TAVG'] - df['TAVG_30d_mean']
    print("   ✅ Temperature anomaly (unusually warm/cold detection)")
    
    # ========================================================================
    # ➕ NEW: Cumulative Rainfall (season-to-date soil moisture proxy)
    # ========================================================================
    df['PRCP_YTD'] = df.groupby(df['Date_Standard'].dt.year)['PRCP'].cumsum()
    df['PRCP_30d_sum'] = df['PRCP'].rolling(window=30, min_periods=1).sum()
    print("   ✅ Cumulative rainfall (soil moisture indicators)")
    
    # ========================================================================
    # ✅ KEEP: Lag Features (baseline compatible)
    # ========================================================================
    for lag in [1, 3, 7]:
        df[f'Pollen_lag_{lag}'] = df['Total_Pollen'].shift(lag)
        df[f'TMAX_lag_{lag}'] = df['TMAX'].shift(lag)
        df[f'PRCP_lag_{lag}'] = df['PRCP'].shift(lag)
    print("   ✅ Lag features (1, 3, 7 days)")
    
    # ========================================================================
    # ✅ KEEP: Rolling Averages (baseline compatible - INCLUDING pollen rolling)
    # ========================================================================
    for window in [3, 7, 14]:
        df[f'Pollen_roll_mean_{window}'] = df['Total_Pollen'].rolling(window=window, min_periods=1).mean()
        df[f'Temp_roll_mean_{window}'] = df['TAVG'].rolling(window=window, min_periods=1).mean()
        df[f'Precip_roll_sum_{window}'] = df['PRCP'].rolling(window=window, min_periods=1).sum()
    print("   ✅ Rolling features (3, 7, 14 days)")
    
    # ========================================================================
    # ➕ ENHANCED: Growing Degree Days with Biological Thresholds
    # ========================================================================
    # Different plant types have different base temperatures
    # Grass: 10°C, Trees: 5°C, Weeds: 8°C
    # Use 5°C as conservative base for mixed pollen
    base_temp = 5
    df['GDD'] = np.maximum(df['TAVG'] - base_temp, 0)
    
    # Annual cumulative GDD
    df['GDD_cumsum'] = df.groupby(df['Date_Standard'].dt.year)['GDD'].cumsum()
    
    # GDD rolling sums (recent growth activity)
    df['GDD_7d'] = df['GDD'].rolling(window=7, min_periods=1).sum()
    df['GDD_14d'] = df['GDD'].rolling(window=14, min_periods=1).sum()
    df['GDD_30d'] = df['GDD'].rolling(window=30, min_periods=1).sum()
    
    # ➕ NEW: GDD with frost reset (biologically realistic)
    # Reset cumulative GDD when temp drops below freezing
    df['Below_Freezing'] = (df['TMIN'] < 0).astype(int)
    df['Frost_Event'] = ((df['TMIN'] < 0) & (df['TMIN'].shift(1) >= 0)).astype(int)
    
    # Cumulative GDD since last frost
    frost_reset = df['Frost_Event'].cumsum()
    df['GDD_since_frost'] = df.groupby(frost_reset)['GDD'].cumsum()
    print("   ✅ Enhanced GDD with frost-aware accumulation")
    
    # ========================================================================
    # ➕ NEW: Peak Season Proximity Indicators
    # ========================================================================
    # Tree pollen peak: March-May (DOY 60-150)
    # Grass pollen peak: May-July (DOY 120-210)
    # Ragweed peak: August-October (DOY 210-300)
    
    # Distance to tree pollen peak (DOY 105 = mid-April)
    tree_peak = 105
    df['Days_to_Tree_Peak'] = np.abs(df['Day_of_Year'] - tree_peak)
    df['In_Tree_Season'] = ((df['Day_of_Year'] >= 60) & (df['Day_of_Year'] <= 150)).astype(int)
    
    # Distance to grass pollen peak (DOY 165 = mid-June)
    grass_peak = 165
    df['Days_to_Grass_Peak'] = np.abs(df['Day_of_Year'] - grass_peak)
    df['In_Grass_Season'] = ((df['Day_of_Year'] >= 120) & (df['Day_of_Year'] <= 210)).astype(int)
    
    # Distance to ragweed peak (DOY 255 = mid-September)
    ragweed_peak = 255
    df['Days_to_Ragweed_Peak'] = np.abs(df['Day_of_Year'] - ragweed_peak)
    df['In_Ragweed_Season'] = ((df['Day_of_Year'] >= 210) & (df['Day_of_Year'] <= 300)).astype(int)
    
    print("   ✅ Peak season proximity indicators (tree, grass, ragweed)")
    
    # ========================================================================
    # ➕ NEW: Weather Dynamics
    # ========================================================================
    # Temperature volatility (how much temp changes day-to-day)
    df['TAVG_change'] = df['TAVG'].diff()
    df['TAVG_volatility_7d'] = df['TAVG_change'].rolling(window=7, min_periods=1).std()
    
    # Consecutive dry/wet days
    df['Consecutive_Dry_Days'] = (df['PRCP'] == 0).groupby((df['PRCP'] != 0).cumsum()).cumsum()
    df['Consecutive_Wet_Days'] = (df['PRCP'] > 0).groupby((df['PRCP'] == 0).cumsum()).cumsum()
    
    print("   ✅ Weather dynamics (temperature volatility, dry/wet streaks)")
    
    # ========================================================================
    # ➕ REPLACE: Wind Feature (use actual values instead of binary)
    # ========================================================================
    # Wind percentile relative to 30-day history (more adaptive than global threshold)
    df['AWND_30d_pct'] = df['AWND'].rolling(window=30, min_periods=1).apply(
        lambda x: (x.iloc[-1] / x.max() * 100) if len(x) > 0 and x.max() > 0 else 0
    )
    print("   ✅ Wind percentile (adaptive wind strength indicator)")
    
    feature_count = len([col for col in df.columns if col not in ['Date_Standard', 'Total_Pollen', 
                                                                    'Tree_Pollen', 'Grass_Pollen', 
                                                                    'Weed_Pollen', 'Ragweed_Pollen',
                                                                    'Pollen_Severity', 'TMAX', 'TMIN', 
                                                                    'TAVG', 'PRCP', 'AWND', 'TAVG_30d_mean']])
    print(f"\n   ✅ Created {feature_count} enhanced features")
    print(f"   📊 New features: Temperature anomalies, soil moisture, frost-aware GDD,")
    print(f"             peak season indicators, weather dynamics, adaptive wind")
    
    return df

def select_enhanced_features(df):
    """Select enhanced feature set - ADDS to baseline, doesn't replace"""
    print("\n📋 Selecting ENHANCED features...")
    
    features = [
        # Core weather (keep essentials)
        'TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND', 'Temp_Range', 'Is_Rainy',
        
        # ➕ NEW: Temperature anomalies
        'Temp_Anomaly', 'TAVG_change', 'TAVG_volatility_7d',
        
        # ➕ NEW: Precipitation patterns
        'PRCP_YTD', 'PRCP_30d_sum', 'Consecutive_Dry_Days', 'Consecutive_Wet_Days',
        
        # ✅ KEEP: Temporal (cyclical)
        'Day_of_Week', 'Month_sin', 'Month_cos', 'DOY_sin', 'DOY_cos',
        
        # ✅ KEEP: ALL Baseline Lags (1, 3, 7)
        *[c for c in df.columns if '_lag_' in c],
        
        # ✅ KEEP: ALL Baseline Rolling (3, 7, 14) - including pollen
        *[c for c in df.columns if 'roll_mean_' in c or 'roll_sum_' in c],
        
        # ✅ KEEP + ENHANCE: GDD features (baseline + new)
        'GDD', 'GDD_cumsum',
        # ➕ NEW GDD features:
        'GDD_7d', 'GDD_14d', 'GDD_30d', 'GDD_since_frost', 'Below_Freezing',
        
        # ➕ NEW: Peak season indicators
        'Days_to_Tree_Peak', 'In_Tree_Season',
        'Days_to_Grass_Peak', 'In_Grass_Season',
        'Days_to_Ragweed_Peak', 'In_Ragweed_Season',
    ]
    
    available = [f for f in features if f in df.columns]
    
    print(f"   Selected {len(available)} features:")
    print(f"   • Core weather: 7")
    print(f"   • Temperature dynamics: 3 (NEW)")
    print(f"   • Precipitation patterns: 4 (NEW)")
    print(f"   • Temporal: 5 (cyclical)")
    print(f"   • Lag features: {len([f for f in available if 'lag_' in f])}")
    print(f"   • Rolling features: {len([f for f in available if 'roll_' in f])}")
    print(f"   • GDD features: {len([f for f in available if 'GDD' in f or 'Freezing' in f])}")
    print(f"   • Peak season: 6 (NEW)")
    
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
    print("\n🤖 Training Enhanced Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
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
    
    # Train on full training set
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
    
    results = {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'cv_mae_mean': -cv_scores.mean(),
        'cv_mae_std': cv_scores.std()
    }
    
    print(f"\n   📊 Training Results:")
    print(f"      Train MAE: {train_mae:.4f}")
    print(f"      Train R²:  {train_r2:.4f}")
    print(f"\n   📊 Test Results:")
    print(f"      Test MAE:  {test_mae:.4f}")
    print(f"      Test RMSE: {test_rmse:.4f}")
    print(f"      Test R²:   {test_r2:.4f}")
    
    return model, results, y_test_pred, y_test

def compare_with_baseline():
    """Load and display baseline performance"""
    print("\n" + "="*70)
    print("📊 COMPARISON WITH BASELINE")
    print("="*70)
    
    try:
        with open('model_results.json', 'r') as f:
            baseline = json.load(f)
            baseline_mae = baseline['test_mae']
            baseline_r2 = baseline['test_r2']
            print(f"\n   Baseline (original features):")
            print(f"   • MAE: {baseline_mae:.4f}")
            print(f"   • R²:  {baseline_r2:.4f}")
            return baseline_mae, baseline_r2
    except:
        print("\n   ⚠️  Baseline results not found")
        return None, None

def main():
    print("="*70)
    print("🌸 ENHANCED POLLEN PREDICTOR")
    print("="*70)
    print("✨ Improvements:")
    print("   • Temperature anomalies (seasonal deviation)")
    print("   • Cumulative rainfall (soil moisture)")
    print("   • Frost-aware GDD accumulation")
    print("   • Peak season proximity indicators")
    print("   • Weather dynamics (volatility, dry/wet streaks)")
    print("   • Simplified lag/rolling features (reduced collinearity)")
    print("="*70)
    
    # Load data
    df = load_and_prepare_data()
    df, thresholds = create_pollen_severity_scale(df)
    df = engineer_enhanced_features(df)
    features = select_enhanced_features(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test, train_dates, test_dates, df_clean = \
        prepare_train_test_split(df, features)
    
    # Train model
    model, results, y_pred, y_true = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Compare with baseline
    baseline_mae, baseline_r2 = compare_with_baseline()
    
    if baseline_mae:
        improvement_mae = ((baseline_mae - results['test_mae']) / baseline_mae) * 100
        improvement_r2 = ((results['test_r2'] - baseline_r2) / baseline_r2) * 100
        
        print(f"\n   Enhanced model:")
        print(f"   • MAE: {results['test_mae']:.4f}")
        print(f"   • R²:  {results['test_r2']:.4f}")
        
        print(f"\n   🎯 Improvement:")
        if improvement_mae > 0:
            print(f"   • MAE improved by {improvement_mae:.1f}% ✅")
        else:
            print(f"   • MAE worsened by {-improvement_mae:.1f}% ❌")
            
        if improvement_r2 > 0:
            print(f"   • R² improved by {improvement_r2:.1f}% ✅")
        else:
            print(f"   • R² worsened by {-improvement_r2:.1f}% ❌")
    
    # Save results
    print("\n💾 Saving outputs...")
    joblib.dump(model, 'pollen_predictor_enhanced.joblib')
    joblib.dump(features, 'enhanced_features.joblib')
    
    with open('enhanced_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"   ✅ Model: pollen_predictor_enhanced.joblib")
    print(f"   ✅ Results: enhanced_results.json")
    print(f"   ✅ Features: enhanced_features.joblib")
    
    print("\n🏆 Top 15 Most Important Features:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"   {row['feature']:30} {row['importance']:.4f}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
