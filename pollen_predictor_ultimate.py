"""
Ultimate Pollen Predictor - All Enhancement Strategies Combined
================================================================

Implements:
1. Pollen momentum features (acceleration, trends, volatility)
2. Weather-pollen interaction features
3. Multi-type pollen models (tree, grass, weed)
4. Stacked ensemble architecture
5. Hyperparameter tuning
6. Direct pollen count prediction (then convert to severity)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint, uniform
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 ULTIMATE POLLEN PREDICTOR")
print("="*80)
print("Implementing ALL enhancement strategies:")
print("  1. ✅ Pollen momentum features (acceleration, trends, volatility)")
print("  2. ✅ Weather-pollen interaction features")
print("  3. ✅ Multi-type pollen models (tree, grass, weed)")
print("  4. ✅ Stacked ensemble architecture")
print("  5. ✅ Hyperparameter tuning")
print("  6. ✅ Direct pollen count prediction")
print("="*80)

def load_data(file_path='combined_allergy_weather.csv'):
    """Load and prepare data"""
    print("\n📂 Loading data...")
    df = pd.read_csv(file_path)
    df['Date_Standard'] = pd.to_datetime(df['Date_Standard'])
    print(f"   Dataset: {df.shape[0]} samples from {df['Date_Standard'].min().date()} to {df['Date_Standard'].max().date()}")
    return df

def create_pollen_severity_scale(df):
    """Create 0-10 pollen severity scale"""
    pollen_data = df['Total_Pollen'].copy()
    percentiles = [0, 10, 25, 40, 55, 70, 80, 85, 90, 95, 100]
    thresholds = np.percentile(pollen_data[pollen_data > 0], percentiles)
    
    def pollen_to_severity(pollen_count):
        if pollen_count == 0:
            return 0
        for i in range(len(thresholds)-1, 0, -1):
            if pollen_count >= thresholds[i]:
                return i
        return 0
    
    df['Pollen_Severity'] = df['Total_Pollen'].apply(pollen_to_severity)
    return df, thresholds

def engineer_ultimate_features(df):
    """
    ULTIMATE FEATURE ENGINEERING
    Combines all enhancement strategies
    """
    print("\n🔬 Engineering ultimate feature set...")
    
    df = df.sort_values('Date_Standard').reset_index(drop=True)
    
    # ========================================================================
    # BASELINE FEATURES (from original model)
    # ========================================================================
    print("   Building baseline features...")
    
    # Temporal
    df['Day_of_Year'] = df['Date_Standard'].dt.dayofyear
    df['Day_of_Week'] = df['Date_Standard'].dt.dayofweek
    df['Month'] = df['Date_Standard'].dt.month
    df['Year'] = df['Date_Standard'].dt.year
    
    # Cyclical
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DOY_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
    df['DOY_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
    
    # Weather basics
    df['Temp_Range'] = df['TMAX'] - df['TMIN']
    df['Is_Rainy'] = (df['PRCP'] > 0).astype(int)
    
    # GDD
    base_temp = 5
    df['GDD'] = np.maximum(df['TAVG'] - base_temp, 0)
    df['GDD_cumsum'] = df.groupby(df['Year'])['GDD'].cumsum()
    
    # Pollen lags
    for lag in [1, 3, 7]:
        df[f'Total_Pollen_lag_{lag}'] = df['Total_Pollen'].shift(lag)
        df[f'Tree_Pollen_lag_{lag}'] = df['Tree'].shift(lag)
        df[f'Grass_Pollen_lag_{lag}'] = df['Grass'].shift(lag)
        df[f'Weed_Pollen_lag_{lag}'] = df['Weed'].shift(lag)
    
    # Weather lags
    for lag in [1, 3, 7]:
        df[f'TMAX_lag_{lag}'] = df['TMAX'].shift(lag)
        df[f'PRCP_lag_{lag}'] = df['PRCP'].shift(lag)
    
    # Pollen rolling
    for window in [3, 7, 14]:
        df[f'Total_Pollen_roll_{window}'] = df['Total_Pollen'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'Tree_Pollen_roll_{window}'] = df['Tree'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'Grass_Pollen_roll_{window}'] = df['Grass'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'Weed_Pollen_roll_{window}'] = df['Weed'].shift(1).rolling(window=window, min_periods=1).mean()
    
    # Weather rolling
    for window in [3, 7, 14]:
        df[f'Temp_roll_{window}'] = df['TAVG'].rolling(window=window, min_periods=1).mean()
        df[f'Precip_roll_{window}'] = df['PRCP'].rolling(window=window, min_periods=1).sum()
    
    print("      ✅ Baseline features complete")
    
    # ========================================================================
    # NEW: POLLEN MOMENTUM FEATURES (Strategy #1)
    # ========================================================================
    print("   Adding pollen momentum features...")
    
    # Acceleration (rate of change)
    df['Pollen_acceleration'] = df['Total_Pollen_lag_1'].diff()
    df['Tree_acceleration'] = df['Tree_Pollen_lag_1'].diff()
    df['Grass_acceleration'] = df['Grass_Pollen_lag_1'].diff()
    df['Weed_acceleration'] = df['Weed_Pollen_lag_1'].diff()
    
    # Trends (short vs long term)
    df['Pollen_trend_3d'] = (df['Total_Pollen_lag_1'] - df['Total_Pollen_lag_3']) / 2
    df['Pollen_trend_7d'] = (df['Total_Pollen_lag_1'] - df['Total_Pollen_lag_7']) / 6
    df['Tree_trend_7d'] = (df['Tree_Pollen_lag_1'] - df['Tree_Pollen_lag_7']) / 6
    df['Grass_trend_7d'] = (df['Grass_Pollen_lag_1'] - df['Grass_Pollen_lag_7']) / 6
    df['Weed_trend_7d'] = (df['Weed_Pollen_lag_1'] - df['Weed_Pollen_lag_7']) / 6
    
    # Volatility (how stable are pollen levels?)
    df['Pollen_volatility_7d'] = df['Total_Pollen_lag_1'].rolling(7, min_periods=1).std()
    df['Tree_volatility_7d'] = df['Tree_Pollen_lag_1'].rolling(7, min_periods=1).std()
    df['Grass_volatility_7d'] = df['Grass_Pollen_lag_1'].rolling(7, min_periods=1).std()
    
    # Spike indicators (sudden jumps)
    df['Pollen_spike'] = (df['Total_Pollen_lag_1'] > df['Total_Pollen_roll_7'] * 1.5).astype(int)
    df['Tree_spike'] = (df['Tree_Pollen_lag_1'] > df['Tree_Pollen_roll_7'] * 1.5).astype(int)
    df['Grass_spike'] = (df['Grass_Pollen_lag_1'] > df['Grass_Pollen_roll_7'] * 1.5).astype(int)
    
    # Momentum direction
    df['Pollen_increasing'] = (df['Pollen_trend_3d'] > 0).astype(int)
    df['Pollen_decreasing'] = (df['Pollen_trend_3d'] < 0).astype(int)
    
    print("      ✅ Momentum features: 18 added")
    
    # ========================================================================
    # NEW: WEATHER-POLLEN INTERACTIONS (Strategy #2)
    # ========================================================================
    print("   Adding weather-pollen interaction features...")
    
    # Rain suppression
    df['Pollen_x_Rain'] = df['Total_Pollen_roll_3'] * df['PRCP']
    df['Pollen_x_Rain_lag1'] = df['Total_Pollen_roll_3'] * df['PRCP_lag_1']
    df['Pollen_suppression_index'] = df['Total_Pollen_roll_3'] * df['Precip_roll_3']
    
    # Wind dispersal
    df['Pollen_x_Wind'] = df['Total_Pollen_roll_3'] * df['AWND']
    df['Tree_x_Wind'] = df['Tree_Pollen_roll_3'] * df['AWND']
    df['Grass_x_Wind'] = df['Grass_Pollen_roll_3'] * df['AWND']
    
    # Temperature effects (affects release rate)
    df['Pollen_x_Temp'] = df['Total_Pollen_roll_3'] * df['TAVG']
    df['Tree_x_Temp'] = df['Tree_Pollen_roll_3'] * df['TAVG']
    df['Grass_x_Temp'] = df['Grass_Pollen_roll_3'] * df['TAVG']
    
    # Dry conditions favor pollen dispersal
    df['Consecutive_Dry_Days'] = (df['PRCP'] == 0).astype(int).groupby((df['PRCP'] > 0).cumsum()).cumsum()
    df['Pollen_x_Dry'] = df['Total_Pollen_roll_3'] * df['Consecutive_Dry_Days']
    
    # Temperature range effects
    df['Pollen_x_TempRange'] = df['Total_Pollen_roll_3'] * df['Temp_Range']
    
    print("      ✅ Interaction features: 12 added")
    
    # ========================================================================
    # ENHANCED BIOLOGICAL FEATURES
    # ========================================================================
    print("   Adding enhanced biological features...")
    
    # Temperature anomaly
    df['TAVG_30d_mean'] = df['TAVG'].rolling(30, min_periods=1).mean()
    df['Temp_Anomaly'] = df['TAVG'] - df['TAVG_30d_mean']
    df['TAVG_change'] = df['TAVG'].diff()
    df['Temp_Volatility_7d'] = df['TAVG_change'].rolling(7, min_periods=1).std()
    
    # Precipitation patterns
    df['PRCP_YTD'] = df.groupby(df['Year'])['PRCP'].cumsum()
    df['PRCP_30d_sum'] = df['PRCP'].rolling(30, min_periods=1).sum()
    
    # Enhanced GDD
    df['GDD_7d'] = df['GDD'].rolling(7, min_periods=1).sum()
    df['GDD_14d'] = df['GDD'].rolling(14, min_periods=1).sum()
    df['GDD_30d'] = df['GDD'].rolling(30, min_periods=1).sum()
    
    # Frost-aware GDD
    df['Below_Freezing'] = (df['TMIN'] < 0).astype(int)
    df['Frost_Event'] = ((df['TMIN'] < 0) & (df['TMIN'].shift(1) >= 0)).astype(int)
    frost_reset = df['Frost_Event'].cumsum()
    df['GDD_since_frost'] = df.groupby(frost_reset)['GDD'].cumsum()
    
    # Peak season indicators
    tree_peak, grass_peak, ragweed_peak = 105, 165, 255
    df['Days_to_Tree_Peak'] = np.abs(df['Day_of_Year'] - tree_peak)
    df['Days_to_Grass_Peak'] = np.abs(df['Day_of_Year'] - grass_peak)
    df['Days_to_Ragweed_Peak'] = np.abs(df['Day_of_Year'] - ragweed_peak)
    df['In_Tree_Season'] = ((df['Day_of_Year'] >= 60) & (df['Day_of_Year'] <= 150)).astype(int)
    df['In_Grass_Season'] = ((df['Day_of_Year'] >= 120) & (df['Day_of_Year'] <= 210)).astype(int)
    df['In_Ragweed_Season'] = ((df['Day_of_Year'] >= 210) & (df['Day_of_Year'] <= 300)).astype(int)
    
    # GDD × Season interactions
    df['GDD_x_Tree_Season'] = df['GDD_cumsum'] * df['In_Tree_Season']
    df['GDD_x_Grass_Season'] = df['GDD_cumsum'] * df['In_Grass_Season']
    df['GDD_x_Ragweed_Season'] = df['GDD_cumsum'] * df['In_Ragweed_Season']
    
    print("      ✅ Biological features: 20 added")
    
    total_features = len([c for c in df.columns if c not in ['Date_Standard', 'Total_Pollen', 
                          'Tree_Pollen', 'Grass_Pollen', 'Weed_Pollen', 'Ragweed_Pollen',
                          'Pollen_Severity', 'Year', 'Month', 'Day_of_Year', 'TAVG_30d_mean',
                          'TAVG_change', 'Frost_Event', 'Consecutive_Dry_Days']])
    print(f"\n   ✅ Total feature count: {total_features}")
    
    return df

def select_features(df):
    """Select comprehensive feature set"""
    print("\n📋 Selecting features...")
    
    # Exclude raw target columns and intermediate calculation columns
    exclude = ['Date_Standard', 'Total_Pollen', 'Tree', 'Grass', 
               'Weed', 'Ragweed', 'Pollen_Severity', 
               'Year', 'Month', 'TAVG_30d_mean', 'TAVG_change', 'Frost_Event',
               'Consecutive_Dry_Days', 'Day_of_Year',
               # Also exclude string/ID columns
               'Date', 'Week', 'Tree_Level', 'Grass_Level', 'Weed_Level', 'Ragweed_Level',
               'OBJECTID', 'Year_Numeric', 'Month_Numeric', 'STATION', 'NAME',
               'Tree_Level_Numeric', 'Grass_Level_Numeric', 'Weed_Level_Numeric', 
               'Ragweed_Level_Numeric', 'TAVG_calculated', 'WSF2', 'WSF5']
    
    features = [c for c in df.columns if c not in exclude and not c.endswith('_lag_0')]
    
    print(f"   Selected {len(features)} features for modeling")
    
    return features

def prepare_data(df, features, target='Pollen_Severity'):
    """Prepare train/test split"""
    print(f"\n📊 Preparing data...")
    
    df_clean = df.dropna(subset=features + [target]).copy()
    print(f"   Clean dataset: {len(df_clean)} samples")
    
    # Time-based split (80/20)
    split_date = df_clean['Date_Standard'].quantile(0.8)
    train_mask = df_clean['Date_Standard'] <= split_date
    test_mask = df_clean['Date_Standard'] > split_date
    
    X_train = df_clean.loc[train_mask, features]
    X_test = df_clean.loc[test_mask, features]
    y_train = df_clean.loc[train_mask, target]
    y_test = df_clean.loc[test_mask, target]
    
    # Also get pollen types for multi-model approach
    y_train_tree = df_clean.loc[train_mask, 'Tree']
    y_train_grass = df_clean.loc[train_mask, 'Grass']
    y_train_weed = df_clean.loc[train_mask, 'Weed']
    y_test_tree = df_clean.loc[test_mask, 'Tree']
    y_test_grass = df_clean.loc[test_mask, 'Grass']
    y_test_weed = df_clean.loc[test_mask, 'Weed']
    y_test_total = df_clean.loc[test_mask, 'Total_Pollen']
    
    train_dates = df_clean.loc[train_mask, 'Date_Standard']
    test_dates = df_clean.loc[test_mask, 'Date_Standard']
    
    print(f"   Train: {len(X_train)} samples ({train_dates.min().date()} to {train_dates.max().date()})")
    print(f"   Test:  {len(X_test)} samples ({test_dates.min().date()} to {test_dates.max().date()})")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_train_tree': y_train_tree, 'y_test_tree': y_test_tree,
        'y_train_grass': y_train_grass, 'y_test_grass': y_test_grass,
        'y_train_weed': y_train_weed, 'y_test_weed': y_test_weed,
        'y_test_total': y_test_total,
        'train_dates': train_dates, 'test_dates': test_dates
    }

def train_tuned_random_forest(X_train, y_train, X_test, y_test, name="Total"):
    """
    Strategy #5: Hyperparameter tuning with RandomizedSearch
    """
    print(f"\n🎯 Training tuned Random Forest for {name} pollen...")
    
    param_distributions = {
        'n_estimators': randint(200, 500),
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', 0.8],
        'bootstrap': [True, False]
    }
    
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=20,  # Try 20 combinations
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"   Running hyperparameter search (20 iterations)...")
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    
    print(f"   Best parameters: {random_search.best_params_}")
    
    # Evaluate
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"   Train MAE: {train_mae:.4f}")
    print(f"   Test MAE:  {test_mae:.4f}, R²: {test_r2:.4f}")
    
    return best_model, y_test_pred

def train_multi_type_models(data):
    """
    Strategy #3: Multi-type pollen models
    Train separate models for tree, grass, weed, then combine
    """
    print("\n" + "="*80)
    print("🌳🌾🌿 STRATEGY #3: Multi-Type Pollen Models")
    print("="*80)
    
    X_train, X_test = data['X_train'], data['X_test']
    
    # Train models for each pollen type
    model_tree, pred_tree = train_tuned_random_forest(
        X_train, data['y_train_tree'], X_test, data['y_test_tree'], "Tree"
    )
    
    model_grass, pred_grass = train_tuned_random_forest(
        X_train, data['y_train_grass'], X_test, data['y_test_grass'], "Grass"
    )
    
    model_weed, pred_weed = train_tuned_random_forest(
        X_train, data['y_train_weed'], X_test, data['y_test_weed'], "Weed"
    )
    
    # Combine predictions
    print("\n   📊 Combining predictions...")
    pred_total = pred_tree + pred_grass + pred_weed
    
    # Convert to severity scale
    # Use percentiles from training data
    y_train_total = data['y_train_tree'] + data['y_train_grass'] + data['y_train_weed']
    percentiles = [0, 10, 25, 40, 55, 70, 80, 85, 90, 95, 100]
    thresholds = np.percentile(y_train_total[y_train_total > 0], percentiles)
    
    def pollen_to_severity(pollen_count):
        if pollen_count == 0:
            return 0
        for i in range(len(thresholds)-1, 0, -1):
            if pollen_count >= thresholds[i]:
                return i
        return 0
    
    pred_severity = np.array([pollen_to_severity(p) for p in pred_total])
    
    # Evaluate
    mae_count = mean_absolute_error(data['y_test_total'], pred_total)
    mae_severity = mean_absolute_error(data['y_test'], pred_severity)
    r2_count = r2_score(data['y_test_total'], pred_total)
    r2_severity = r2_score(data['y_test'], pred_severity)
    
    print(f"\n   ✅ Multi-Type Model Results:")
    print(f"      Pollen Count - MAE: {mae_count:.2f}, R²: {r2_count:.4f}")
    print(f"      Severity (0-10) - MAE: {mae_severity:.4f}, R²: {r2_severity:.4f}")
    
    return {
        'models': {'tree': model_tree, 'grass': model_grass, 'weed': model_weed},
        'predictions': pred_severity,
        'mae': mae_severity,
        'r2': r2_severity
    }

def train_stacked_ensemble(data):
    """
    Strategy #4: Stacked ensemble architecture
    Level 1: RF, GradientBoosting, Ridge
    Level 2: Meta-learner combines predictions
    """
    print("\n" + "="*80)
    print("🏗️  STRATEGY #4: Stacked Ensemble")
    print("="*80)
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Level 1 Models
    print("\n   Training Level 1 models...")
    
    # Model 1: Random Forest
    print("      1/3: Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    print(f"          Test MAE: {mean_absolute_error(y_test, rf_test_pred):.4f}")
    
    # Model 2: Gradient Boosting
    print("      2/3: Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_train_pred = gb_model.predict(X_train)
    gb_test_pred = gb_model.predict(X_test)
    print(f"          Test MAE: {mean_absolute_error(y_test, gb_test_pred):.4f}")
    
    # Model 3: Ridge Regression
    print("      3/3: Ridge Regression...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_train_pred = ridge_model.predict(X_train)
    ridge_test_pred = ridge_model.predict(X_test)
    print(f"          Test MAE: {mean_absolute_error(y_test, ridge_test_pred):.4f}")
    
    # Level 2: Meta-learner
    print("\n   Training Level 2 meta-learner...")
    meta_train = np.column_stack([rf_train_pred, gb_train_pred, ridge_train_pred])
    meta_test = np.column_stack([rf_test_pred, gb_test_pred, ridge_test_pred])
    
    meta_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    meta_model.fit(meta_train, y_train)
    
    ensemble_pred = meta_model.predict(meta_test)
    
    mae = mean_absolute_error(y_test, ensemble_pred)
    r2 = r2_score(y_test, ensemble_pred)
    
    print(f"\n   ✅ Stacked Ensemble Results:")
    print(f"      MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Meta-learner weights
    print(f"\n   Meta-learner feature importance (model weights):")
    print(f"      Random Forest: {meta_model.feature_importances_[0]:.3f}")
    print(f"      Gradient Boost: {meta_model.feature_importances_[1]:.3f}")
    print(f"      Ridge: {meta_model.feature_importances_[2]:.3f}")
    
    return {
        'models': {'rf': rf_model, 'gb': gb_model, 'ridge': ridge_model, 'meta': meta_model},
        'predictions': ensemble_pred,
        'mae': mae,
        'r2': r2
    }

def train_direct_pollen_model(data, features):
    """
    Strategy #6: Direct pollen count prediction (then convert to severity)
    """
    print("\n" + "="*80)
    print("🎯 STRATEGY #6: Direct Pollen Count Prediction")
    print("="*80)
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train_count = data['y_train_tree'] + data['y_train_grass'] + data['y_train_weed']
    y_test_count = data['y_test_total']
    
    print("\n   Training Random Forest on pollen counts...")
    
    param_distributions = {
        'n_estimators': randint(200, 500),
        'max_depth': [20, 25, 30, None],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', 0.8],
    }
    
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    
    random_search = RandomizedSearchCV(
        base_model, param_distributions, n_iter=20, cv=tscv,
        scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1, verbose=0
    )
    
    random_search.fit(X_train, y_train_count)
    model = random_search.best_estimator_
    
    # Predict counts
    pred_count = model.predict(X_test)
    pred_count = np.maximum(pred_count, 0)  # No negative pollen
    
    # Convert to severity
    percentiles = [0, 10, 25, 40, 55, 70, 80, 85, 90, 95, 100]
    thresholds = np.percentile(y_train_count[y_train_count > 0], percentiles)
    
    def pollen_to_severity(pollen_count):
        if pollen_count == 0:
            return 0
        for i in range(len(thresholds)-1, 0, -1):
            if pollen_count >= thresholds[i]:
                return i
        return 0
    
    pred_severity = np.array([pollen_to_severity(p) for p in pred_count])
    
    mae_count = mean_absolute_error(y_test_count, pred_count)
    r2_count = r2_score(y_test_count, pred_count)
    mae_severity = mean_absolute_error(data['y_test'], pred_severity)
    r2_severity = r2_score(data['y_test'], pred_severity)
    
    print(f"\n   ✅ Direct Count Model Results:")
    print(f"      Pollen Count - MAE: {mae_count:.2f}, R²: {r2_count:.4f}")
    print(f"      Severity (0-10) - MAE: {mae_severity:.4f}, R²: {r2_severity:.4f}")
    
    return {
        'model': model,
        'predictions': pred_severity,
        'mae': mae_severity,
        'r2': r2_severity
    }

def main():
    # Load data
    df = load_data()
    df, thresholds = create_pollen_severity_scale(df)
    
    # Engineer ultimate features
    df = engineer_ultimate_features(df)
    features = select_features(df)
    
    # Prepare data
    data = prepare_data(df, features)
    
    # Train baseline for comparison
    print("\n" + "="*80)
    print("📊 BASELINE: Standard Random Forest")
    print("="*80)
    baseline_model, _ = train_tuned_random_forest(
        data['X_train'], data['y_train'], data['X_test'], data['y_test'], "Baseline"
    )
    baseline_pred = baseline_model.predict(data['X_test'])
    baseline_mae = mean_absolute_error(data['y_test'], baseline_pred)
    baseline_r2 = r2_score(data['y_test'], baseline_pred)
    
    # Strategy #3: Multi-type models
    multitype_results = train_multi_type_models(data)
    
    # Strategy #4: Stacked ensemble
    ensemble_results = train_stacked_ensemble(data)
    
    # Strategy #6: Direct pollen count
    direct_results = train_direct_pollen_model(data, features)
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS COMPARISON")
    print("="*80)
    
    results = {
        'Baseline (Tuned RF)': {'mae': baseline_mae, 'r2': baseline_r2},
        'Multi-Type Models': {'mae': multitype_results['mae'], 'r2': multitype_results['r2']},
        'Stacked Ensemble': {'mae': ensemble_results['mae'], 'r2': ensemble_results['r2']},
        'Direct Count Model': {'mae': direct_results['mae'], 'r2': direct_results['r2']}
    }
    
    print(f"\n{'Method':<25} {'MAE':<12} {'R²':<12} {'vs Baseline'}")
    print("-" * 70)
    
    for method, metrics in results.items():
        mae, r2 = metrics['mae'], metrics['r2']
        if method == 'Baseline (Tuned RF)':
            improvement = "-"
        else:
            diff = ((baseline_mae - mae) / baseline_mae) * 100
            improvement = f"{diff:+.1f}%"
        print(f"{method:<25} {mae:<12.4f} {r2:<12.4f} {improvement}")
    
    # Find best model
    best_method = min(results.keys(), key=lambda k: results[k]['mae'] if k != 'Baseline (Tuned RF)' else float('inf'))
    best_mae = results[best_method]['mae']
    improvement = ((baseline_mae - best_mae) / baseline_mae) * 100
    
    print("\n" + "="*80)
    print(f"🎯 BEST MODEL: {best_method}")
    print(f"   MAE: {best_mae:.4f} (improvement: {improvement:+.1f}%)")
    print("="*80)
    
    # Save results
    print("\n💾 Saving outputs...")
    with open('ultimate_results.json', 'w') as f:
        json.dump({k: {'mae': float(v['mae']), 'r2': float(v['r2'])} for k, v in results.items()}, f, indent=2)
    
    # Save best model
    if best_method == 'Multi-Type Models':
        joblib.dump(multitype_results['models'], 'pollen_predictor_best.joblib')
    elif best_method == 'Stacked Ensemble':
        joblib.dump(ensemble_results['models'], 'pollen_predictor_best.joblib')
    elif best_method == 'Direct Count Model':
        joblib.dump(direct_results['model'], 'pollen_predictor_best.joblib')
    else:
        joblib.dump(baseline_model, 'pollen_predictor_best.joblib')
    
    joblib.dump(features, 'ultimate_features.joblib')
    
    print("   ✅ Best model: pollen_predictor_best.joblib")
    print("   ✅ Features: ultimate_features.joblib")
    print("   ✅ Results: ultimate_results.json")
    
    print("\n✅ ULTIMATE TRAINING COMPLETE!")

if __name__ == "__main__":
    main()
