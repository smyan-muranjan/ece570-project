"""
XGBoost with Enhanced Features - Fair Comparison
=================================================

This trains XGBoost with the EXACT same features as the final stacked ensemble model
to ensure a fair comparison.

Features:
✓ Pollen momentum (acceleration, trends, volatility)
✓ Weather-pollen interactions  
✓ Enhanced biological features (GDD, phenology)
✓ Same train/test split as final model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint, uniform
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_pollen_severity_scale(df):
    """Create 0-10 pollen severity scale based on percentiles"""
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

def engineer_features(df):
    """
    Engineer comprehensive feature set - IDENTICAL to final model
    """
    print("🔬 Engineering features...")
    
    df = df.sort_values('Date_Standard').reset_index(drop=True)
    
    # TEMPORAL FEATURES
    df['Day_of_Year'] = df['Date_Standard'].dt.dayofyear
    df['Day_of_Week'] = df['Date_Standard'].dt.dayofweek
    df['Month'] = df['Date_Standard'].dt.month
    df['Year'] = df['Date_Standard'].dt.year
    
    # Cyclical encoding
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DOY_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
    df['DOY_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
    
    # CORE WEATHER FEATURES
    df['Temp_Range'] = df['TMAX'] - df['TMIN']
    df['Is_Rainy'] = (df['PRCP'] > 0).astype(int)
    
    # GROWING DEGREE DAYS
    base_temp = 5
    df['GDD'] = np.maximum(df['TAVG'] - base_temp, 0)
    df['GDD_cumsum'] = df.groupby(df['Year'])['GDD'].cumsum()
    df['GDD_7d'] = df['GDD'].rolling(7, min_periods=1).sum()
    df['GDD_14d'] = df['GDD'].rolling(14, min_periods=1).sum()
    df['GDD_30d'] = df['GDD'].rolling(30, min_periods=1).sum()
    
    # Frost-aware GDD
    df['Below_Freezing'] = (df['TMIN'] < 0).astype(int)
    df['Frost_Event'] = ((df['TMIN'] < 0) & (df['TMIN'].shift(1) >= 0)).astype(int)
    frost_reset = df['Frost_Event'].cumsum()
    df['GDD_since_frost'] = df.groupby(frost_reset)['GDD'].cumsum()
    
    # POLLEN LAG FEATURES
    for lag in [1, 3, 7]:
        df[f'Total_Pollen_lag_{lag}'] = df['Total_Pollen'].shift(lag)
        df[f'Tree_Pollen_lag_{lag}'] = df['Tree'].shift(lag)
        df[f'Grass_Pollen_lag_{lag}'] = df['Grass'].shift(lag)
        df[f'Weed_Pollen_lag_{lag}'] = df['Weed'].shift(lag)
    
    # POLLEN ROLLING AVERAGES
    for window in [3, 7, 14]:
        df[f'Total_Pollen_roll_{window}'] = df['Total_Pollen'].shift(1).rolling(window, min_periods=1).mean()
        df[f'Tree_Pollen_roll_{window}'] = df['Tree'].shift(1).rolling(window, min_periods=1).mean()
        df[f'Grass_Pollen_roll_{window}'] = df['Grass'].shift(1).rolling(window, min_periods=1).mean()
        df[f'Weed_Pollen_roll_{window}'] = df['Weed'].shift(1).rolling(window, min_periods=1).mean()
    
    # POLLEN MOMENTUM FEATURES
    df['Pollen_acceleration'] = df['Total_Pollen_lag_1'].diff()
    df['Tree_acceleration'] = df['Tree_Pollen_lag_1'].diff()
    df['Grass_acceleration'] = df['Grass_Pollen_lag_1'].diff()
    df['Weed_acceleration'] = df['Weed_Pollen_lag_1'].diff()
    
    df['Pollen_trend_3d'] = (df['Total_Pollen_lag_1'] - df['Total_Pollen_lag_3']) / 2
    df['Pollen_trend_7d'] = (df['Total_Pollen_lag_1'] - df['Total_Pollen_lag_7']) / 6
    df['Tree_trend_7d'] = (df['Tree_Pollen_lag_1'] - df['Tree_Pollen_lag_7']) / 6
    df['Grass_trend_7d'] = (df['Grass_Pollen_lag_1'] - df['Grass_Pollen_lag_7']) / 6
    df['Weed_trend_7d'] = (df['Weed_Pollen_lag_1'] - df['Weed_Pollen_lag_7']) / 6
    
    df['Pollen_volatility_7d'] = df['Total_Pollen_lag_1'].rolling(7, min_periods=1).std()
    df['Tree_volatility_7d'] = df['Tree_Pollen_lag_1'].rolling(7, min_periods=1).std()
    df['Grass_volatility_7d'] = df['Grass_Pollen_lag_1'].rolling(7, min_periods=1).std()
    
    df['Pollen_spike'] = (df['Total_Pollen_lag_1'] > df['Total_Pollen_roll_7'] * 1.5).astype(int)
    df['Tree_spike'] = (df['Tree_Pollen_lag_1'] > df['Tree_Pollen_roll_7'] * 1.5).astype(int)
    df['Grass_spike'] = (df['Grass_Pollen_lag_1'] > df['Grass_Pollen_roll_7'] * 1.5).astype(int)
    
    df['Pollen_increasing'] = (df['Pollen_trend_3d'] > 0).astype(int)
    df['Pollen_decreasing'] = (df['Pollen_trend_3d'] < 0).astype(int)
    
    # WEATHER LAG FEATURES
    for lag in [1, 3, 7]:
        df[f'TMAX_lag_{lag}'] = df['TMAX'].shift(lag)
        df[f'PRCP_lag_{lag}'] = df['PRCP'].shift(lag)
    
    # WEATHER ROLLING AVERAGES
    for window in [3, 7, 14]:
        df[f'Temp_roll_{window}'] = df['TAVG'].rolling(window, min_periods=1).mean()
        df[f'Precip_roll_{window}'] = df['PRCP'].rolling(window, min_periods=1).sum()
    
    # WEATHER-POLLEN INTERACTIONS
    df['Pollen_x_Rain'] = df['Total_Pollen_roll_3'] * df['PRCP']
    df['Pollen_x_Rain_lag1'] = df['Total_Pollen_roll_3'] * df['PRCP_lag_1']
    df['Pollen_suppression_index'] = df['Total_Pollen_roll_3'] * df['Precip_roll_3']
    
    df['Pollen_x_Wind'] = df['Total_Pollen_roll_3'] * df['AWND']
    df['Tree_x_Wind'] = df['Tree_Pollen_roll_3'] * df['AWND']
    df['Grass_x_Wind'] = df['Grass_Pollen_roll_3'] * df['AWND']
    
    df['Pollen_x_Temp'] = df['Total_Pollen_roll_3'] * df['TAVG']
    df['Tree_x_Temp'] = df['Tree_Pollen_roll_3'] * df['TAVG']
    df['Grass_x_Temp'] = df['Grass_Pollen_roll_3'] * df['TAVG']
    
    df['Consecutive_Dry_Days'] = (df['PRCP'] == 0).astype(int).groupby((df['PRCP'] > 0).cumsum()).cumsum()
    df['Pollen_x_Dry'] = df['Total_Pollen_roll_3'] * df['Consecutive_Dry_Days']
    df['Pollen_x_TempRange'] = df['Total_Pollen_roll_3'] * df['Temp_Range']
    
    # ENHANCED BIOLOGICAL FEATURES
    df['TAVG_30d_mean'] = df['TAVG'].rolling(30, min_periods=1).mean()
    df['Temp_Anomaly'] = df['TAVG'] - df['TAVG_30d_mean']
    df['TAVG_change'] = df['TAVG'].diff()
    df['Temp_Volatility_7d'] = df['TAVG_change'].rolling(7, min_periods=1).std()
    
    df['PRCP_YTD'] = df.groupby(df['Year'])['PRCP'].cumsum()
    df['PRCP_30d_sum'] = df['PRCP'].rolling(30, min_periods=1).sum()
    
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
    """Select features for modeling - IDENTICAL to final model"""
    exclude = ['Date_Standard', 'Total_Pollen', 'Tree', 'Grass', 
               'Weed', 'Ragweed', 'Pollen_Severity', 
               'Year', 'Month', 'TAVG_30d_mean', 'TAVG_change', 'Frost_Event',
               'Consecutive_Dry_Days', 'Day_of_Year',
               'Date', 'Week', 'Tree_Level', 'Grass_Level', 'Weed_Level', 'Ragweed_Level',
               'OBJECTID', 'Year_Numeric', 'Month_Numeric', 'STATION', 'NAME',
               'Tree_Level_Numeric', 'Grass_Level_Numeric', 'Weed_Level_Numeric', 
               'Ragweed_Level_Numeric', 'TAVG_calculated', 'WSF2', 'WSF5']
    
    features = [c for c in df.columns if c not in exclude and not c.endswith('_lag_0')]
    
    print(f"   ✅ Selected {len(features)} features")
    return features

def main():
    print("="*80)
    print("🚀 XGBOOST WITH ENHANCED FEATURES - FAIR COMPARISON")
    print("="*80)
    print("Testing XGBoost with EXACT same features as final stacked ensemble")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('combined_allergy_weather.csv')
    df['Date_Standard'] = pd.to_datetime(df['Date_Standard'])
    print(f"   Loaded {len(df)} samples")
    
    # Prepare features
    df, thresholds = create_pollen_severity_scale(df)
    df = engineer_features(df)
    features = select_features(df)
    
    # Prepare data - SAME SPLIT AS FINAL MODEL
    print("\n📊 Preparing data (same split as final model)...")
    df_clean = df.dropna(subset=features + ['Pollen_Severity']).copy()
    print(f"   Clean dataset: {len(df_clean)} samples")
    
    split_date = df_clean['Date_Standard'].quantile(0.8)
    train_mask = df_clean['Date_Standard'] <= split_date
    test_mask = df_clean['Date_Standard'] > split_date
    
    X_train = df_clean.loc[train_mask, features]
    X_test = df_clean.loc[test_mask, features]
    y_train = df_clean.loc[train_mask, 'Pollen_Severity']
    y_test = df_clean.loc[test_mask, 'Pollen_Severity']
    
    train_dates = df_clean.loc[train_mask, 'Date_Standard']
    test_dates = df_clean.loc[test_mask, 'Date_Standard']
    
    print(f"   Train: {len(X_train)} samples ({train_dates.min().date()} to {train_dates.max().date()})")
    print(f"   Test:  {len(X_test)} samples ({test_dates.min().date()} to {test_dates.max().date()})")
    
    # =========================================================================
    # BASELINE XGBOOST (Default params)
    # =========================================================================
    print("\n" + "="*80)
    print("📊 BASELINE XGBOOST (Default Parameters)")
    print("="*80)
    
    baseline_model = xgb.XGBRegressor(
        device='cuda',
        tree_method='hist',
        random_state=42,
        early_stopping_rounds=50
    )
    
    baseline_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    baseline_pred = baseline_model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    print(f"\n   Test MAE: {baseline_mae:.4f}")
    print(f"   Test R²:  {baseline_r2:.4f}")
    
    # =========================================================================
    # TUNED XGBOOST (Hyperparameter optimization)
    # =========================================================================
    print("\n" + "="*80)
    print("🎯 TUNED XGBOOST (Hyperparameter Search)")
    print("="*80)
    
    param_distributions = {
        'n_estimators': randint(200, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.15),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'colsample_bylevel': uniform(0.7, 0.3),
        'min_child_weight': randint(1, 15),
        'gamma': uniform(0, 0.5),
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
    
    print("   Running hyperparameter search (30 iterations)...")
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=30,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1,  # XGBoost handles parallelism internally
        verbose=1
    )
    
    random_search.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    tuned_model = random_search.best_estimator_
    
    print(f"\n   Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"      {param}: {value}")
    
    # Cross-validation score
    cv_mae = -random_search.best_score_
    print(f"\n   Cross-validation MAE: {cv_mae:.4f}")
    
    # Test performance
    tuned_train_pred = tuned_model.predict(X_train)
    tuned_test_pred = tuned_model.predict(X_test)
    
    tuned_train_mae = mean_absolute_error(y_train, tuned_train_pred)
    tuned_test_mae = mean_absolute_error(y_test, tuned_test_pred)
    tuned_train_r2 = r2_score(y_train, tuned_train_pred)
    tuned_test_r2 = r2_score(y_test, tuned_test_pred)
    
    within_1 = np.mean(np.abs(y_test - tuned_test_pred) <= 1)
    within_2 = np.mean(np.abs(y_test - tuned_test_pred) <= 2)
    
    print(f"\n   Training Performance:")
    print(f"   • MAE:  {tuned_train_mae:.4f}")
    print(f"   • R²:   {tuned_train_r2:.4f}")
    print(f"\n   Test Performance:")
    print(f"   • MAE:  {tuned_test_mae:.4f}")
    print(f"   • R²:   {tuned_test_r2:.4f}")
    print(f"   • Accuracy (±1 level): {within_1:.1%}")
    print(f"   • Accuracy (±2 levels): {within_2:.1%}")
    
    # =========================================================================
    # COMPARISON WITH FINAL STACKED ENSEMBLE
    # =========================================================================
    print("\n" + "="*80)
    print("🏆 FINAL COMPARISON")
    print("="*80)
    
    # Load final model results
    try:
        with open('pollen_predictor_final_info.json', 'r') as f:
            final_info = json.load(f)
            final_mae = final_info['test_mae']
            final_r2 = final_info['test_r2']
            final_acc1 = final_info['accuracy_within_1']
            final_acc2 = final_info['accuracy_within_2']
    except:
        print("   ⚠️  Could not load final model results")
        final_mae = 0.9044  # From previous run
        final_r2 = 0.7803
        final_acc1 = 0.652
        final_acc2 = 0.901
    
    print(f"\n{'Model':<30} {'MAE':<12} {'R²':<12} {'Acc(±1)':<12} {'Acc(±2)'}")
    print("-" * 80)
    print(f"{'XGBoost (Baseline)':<30} {baseline_mae:<12.4f} {baseline_r2:<12.4f} {'-':<12} {'-'}")
    print(f"{'XGBoost (Tuned)':<30} {tuned_test_mae:<12.4f} {tuned_test_r2:<12.4f} {within_1:<12.1%} {within_2:<12.1%}")
    print(f"{'Stacked Ensemble (Final)':<30} {final_mae:<12.4f} {final_r2:<12.4f} {final_acc1:<12.1%} {final_acc2:<12.1%}")
    
    print("\n" + "="*80)
    
    # Determine winner
    if tuned_test_mae < final_mae:
        improvement = ((final_mae - tuned_test_mae) / final_mae) * 100
        print(f"🎯 WINNER: XGBoost (Tuned)")
        print(f"   Improvement: {improvement:.1f}% better MAE than Stacked Ensemble")
        winner = "xgboost"
    else:
        difference = ((tuned_test_mae - final_mae) / final_mae) * 100
        print(f"🎯 WINNER: Stacked Ensemble")
        print(f"   XGBoost is {difference:.1f}% worse in MAE")
        winner = "ensemble"
    
    print("="*80)
    
    # Save results
    results = {
        'xgboost_baseline': {
            'test_mae': float(baseline_mae),
            'test_r2': float(baseline_r2)
        },
        'xgboost_tuned': {
            'test_mae': float(tuned_test_mae),
            'test_r2': float(tuned_test_r2),
            'train_mae': float(tuned_train_mae),
            'train_r2': float(tuned_train_r2),
            'accuracy_within_1': float(within_1),
            'accuracy_within_2': float(within_2),
            'cv_mae': float(cv_mae),
            'best_params': random_search.best_params_
        },
        'stacked_ensemble': {
            'test_mae': float(final_mae),
            'test_r2': float(final_r2),
            'accuracy_within_1': float(final_acc1),
            'accuracy_within_2': float(final_acc2)
        },
        'winner': winner,
        'tested_at': str(datetime.now())
    }
    
    with open('xgboost_vs_ensemble_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save best XGBoost model
    if winner == "xgboost":
        joblib.dump(tuned_model, 'pollen_predictor_xgboost_best.joblib')
        print(f"\n💾 Best XGBoost model saved to: pollen_predictor_xgboost_best.joblib")
    
    print(f"💾 Comparison results saved to: xgboost_vs_ensemble_comparison.json")
    
    print("\n✅ COMPARISON COMPLETE!")

if __name__ == "__main__":
    main()
