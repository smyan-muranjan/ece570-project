"""
Multi-Type XGBoost Pollen Predictor with GPU Acceleration
- AAAAI biological thresholds for each pollen type
- Refined feature engineering (weather dynamics + peak proximity)
- GPU-accelerated training on RTX 2080
- Separate models for Tree, Grass, Weed, Ragweed, Total_Pollen
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
    HAS_XGB = True
except ImportError:
    print("❌ XGBoost not installed! Install with: pip install xgboost")
    exit(1)

from datetime import datetime
import joblib
import json
import sys
import argparse


# ============================================================================
# AAAAI BIOLOGICAL THRESHOLDS
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
        
        # Extend thresholds to 0-10 scale
        if len(thresholds) == 5:  # Standard AAAAI
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
# REFINED FEATURE ENGINEERING
# ============================================================================

def engineer_refined_features(df):
    """Refined feature engineering with weather dynamics & peak proximity"""
    print("\n🔧 Engineering refined features...")
    
    df = df.sort_values('Date_Standard').reset_index(drop=True)
    
    # ========== CORE TEMPORAL (Cyclical only) ==========
    df['Day_of_Week'] = df['Date_Standard'].dt.dayofweek
    df['Month_Numeric'] = df['Date_Standard'].dt.month
    df['Year_Numeric'] = df['Date_Standard'].dt.year
    df['Day_of_Year'] = df['Date_Standard'].dt.dayofyear
    
    # Cyclical encoding (superior to raw values)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month_Numeric'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month_Numeric'] / 12)
    df['DOY_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
    df['DOY_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
    
    # ========== CORE WEATHER ==========
    df['Temp_Range'] = df['TMAX'] - df['TMIN']
    df['Is_Rainy'] = (df['PRCP'] > 0).astype(int)
    
    # ========== WEATHER DYNAMICS (NEW) ==========
    # Temperature anomaly
    df['TAVG_30day'] = df['TAVG'].rolling(window=30, min_periods=7).mean()
    df['Temp_Anomaly'] = df['TAVG'] - df['TAVG_30day']
    
    # Season-to-date rainfall
    df['Season'] = df['Month_Numeric'].map(lambda m: 
        'Spring' if m in [3, 4, 5] else
        'Summer' if m in [6, 7, 8] else
        'Fall' if m in [9, 10, 11] else 'Winter'
    )
    df['Rainfall_Season_Cumsum'] = df.groupby([df['Year_Numeric'], df['Season']])['PRCP'].cumsum()
    
    # Wind percentile (better than binary threshold)
    df['Wind_Percentile'] = df['AWND'].rolling(window=30, min_periods=7).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50
    )
    
    # ========== LAG FEATURES ==========
    for lag in [1, 3, 7]:
        df[f'Pollen_lag_{lag}'] = df['Total_Pollen'].shift(lag)
        if lag <= 3:  # Reduce redundancy
            df[f'TMAX_lag_{lag}'] = df['TMAX'].shift(lag)
            df[f'PRCP_lag_{lag}'] = df['PRCP'].shift(lag)
    
    # ========== ROLLING (Simplified: 3, 7 only) ==========
    for window in [3, 7]:
        df[f'Pollen_roll_{window}'] = df['Total_Pollen'].rolling(window, min_periods=1).mean()
        df[f'Temp_roll_{window}'] = df['TAVG'].rolling(window, min_periods=1).mean()
        df[f'Rain_roll_{window}'] = df['PRCP'].rolling(window, min_periods=1).sum()
    
    # ========== GROWING DEGREE DAYS (Species-specific) ==========
    df['GDD_General'] = np.maximum(df['TAVG'] - 5, 0)
    df['GDD_Tree'] = np.maximum(df['TAVG'] - 5, 0)  # 5°C base
    df['GDD_Grass'] = np.maximum(df['TAVG'] - 10, 0)  # 10°C base
    df['GDD_Weed'] = np.maximum(df['TAVG'] - 8, 0)  # 8°C base
    
    df['GDD_cumsum'] = df.groupby(df['Year_Numeric'])['GDD_General'].cumsum()
    df['GDD_Tree_cumsum'] = df.groupby(df['Year_Numeric'])['GDD_Tree'].cumsum()
    df['GDD_Grass_cumsum'] = df.groupby(df['Year_Numeric'])['GDD_Grass'].cumsum()
    df['GDD_Weed_cumsum'] = df.groupby(df['Year_Numeric'])['GDD_Weed'].cumsum()
    
    # ========== PEAK SEASON PROXIMITY (NEW) ==========
    # Gaussian proximity to known pollen peaks
    df['Tree_Peak_Prox'] = np.exp(-0.5 * ((df['Day_of_Year'] - 110) / 30) ** 2)  # DOY 110 ≈ mid-April
    df['Grass_Peak_Prox'] = np.exp(-0.5 * ((df['Day_of_Year'] - 175) / 30) ** 2)  # DOY 175 ≈ late June
    df['Weed_Peak_Prox'] = np.exp(-0.5 * ((df['Day_of_Year'] - 255) / 30) ** 2)  # DOY 255 ≈ mid-Sept
    
    # ========== INTERACTION FEATURES ==========
    df['Temp_x_Spring'] = df['TAVG'] * (df['Month_Numeric'].isin([3, 4, 5])).astype(int)
    df['Temp_x_Summer'] = df['TAVG'] * (df['Month_Numeric'].isin([6, 7, 8])).astype(int)
    df['Rain_x_Wind'] = df['PRCP'] * df['AWND']
    
    print(f"   ✅ Created {len([c for c in df.columns if any(x in c for x in ['lag_', 'roll_', 'sin', 'cos', 'GDD', 'Prox', '_x_'])])} features")
    
    return df


def select_refined_features(df):
    """Select refined feature set (removed redundant features)"""
    print("\n📋 Selecting refined features...")
    
    features = [
        # Core weather
        'TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND', 'Temp_Range', 'Is_Rainy',
        # Weather dynamics
        'Temp_Anomaly', 'Rainfall_Season_Cumsum', 'Wind_Percentile',
        # Temporal (cyclical only, keep Day_of_Week)
        'Day_of_Week', 'Month_sin', 'Month_cos', 'DOY_sin', 'DOY_cos',
        'Year_Numeric', 'Month_Numeric',
        # Lags
        *[c for c in df.columns if 'lag_' in c],
        # Rolling
        *[c for c in df.columns if 'roll_' in c and '30day' not in c],
        # GDD
        *[c for c in df.columns if 'GDD' in c and '30day' not in c],
        # Peak proximity
        *[c for c in df.columns if 'Peak_Prox' in c],
        # Interactions
        *[c for c in df.columns if '_x_' in c]
    ]
    
    available = [f for f in features if f in df.columns]
    print(f"   ✅ Selected {len(available)} refined features")
    
    return available


# ============================================================================
# GPU-ACCELERATED XGBOOST
# ============================================================================

def get_gpu_xgboost_config():
    """XGBoost config optimized for RTX 2080 GPU (XGBoost 3.1+ API)"""
    return {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # Use hist with device='cuda' for GPU
        'device': 'cuda',  # GPU acceleration (XGBoost 3.1+)
        'n_estimators': 500,  # Increased, will use early stopping
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 50  # Stop if no improvement for 50 rounds
    }


def get_hyperparameter_search_space():
    """Hyperparameter search space for tuning"""
    return {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.03, 0.05, 0.07],
        'max_depth': [5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0.5, 1.0, 2.0]
    }


# ============================================================================
# METRICS
# ============================================================================

def comprehensive_metrics(y_true, y_pred):
    """Calculate all metrics"""
    high_mask = y_true >= 7
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'weighted_mae': np.mean(np.abs(y_true - y_pred) * np.where(y_true >= 7, 2.0, 1.0)),
        'acc_within_1': np.mean(np.abs(y_true - y_pred) <= 1),
        'acc_within_2': np.mean(np.abs(y_true - y_pred) <= 2),
        'high_mae': mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if np.any(high_mask) else np.nan,
        'high_count': int(np.sum(high_mask))
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_xgboost_for_pollen_type(X_train, y_train, X_test, y_test, pollen_type, 
                                   use_gpu=True, tune=False):
    """Train XGBoost for a specific pollen type"""
    import time
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING: {pollen_type}")
    print(f"{'='*70}")
    print(f"📊 Dataset: {len(X_train):,} train samples, {len(X_test):,} test samples")
    print(f"📋 Features: {X_train.shape[1]}")
    
    # Check GPU availability
    print(f"\n🔍 Checking GPU availability...")
    if use_gpu:
        try:
            gpu_test_start = time.time()
            test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
            test_model.fit(X_train[:100], y_train[:100])
            gpu_test_time = time.time() - gpu_test_start
            print(f"✅ RTX 2080 GPU detected and working! (test: {gpu_test_time:.2f}s)")
            config = get_gpu_xgboost_config()
            device_name = "GPU (CUDA)"
        except Exception as e:
            print(f"⚠️  GPU not available ({str(e)[:60]}...)")
            print(f"   Falling back to CPU")
            config = get_gpu_xgboost_config()
            config['device'] = 'cpu'
            device_name = "CPU"
    else:
        config = get_gpu_xgboost_config()
        config['device'] = 'cpu'
        device_name = "CPU"
    
    print(f"🖥️  Training device: {device_name}")
    print(f"🌳 Trees: {config['n_estimators']}, Max depth: {config['max_depth']}, LR: {config['learning_rate']}")
    
    if tune:
        print(f"\n🔍 Hyperparameter tuning (30 iterations, 5-fold CV)...")
        print(f"   This will train {30 * 5} = 150 models...")
        
        tune_start = time.time()
        base = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method=config['tree_method'],
            device=config['device'],
            random_state=42,
            verbosity=0
        )
        
        search = RandomizedSearchCV(
            base, get_hyperparameter_search_space(),
            n_iter=30, cv=TimeSeriesSplit(n_splits=5),
            scoring='neg_mean_absolute_error',
            random_state=42, n_jobs=1, verbose=2  # verbose=2 for progress
        )
        
        print(f"   ⏱️  Starting hyperparameter search...")
        search.fit(X_train, y_train)
        tune_time = time.time() - tune_start
        
        model = search.best_estimator_
        
        print(f"\n✅ Tuning complete in {tune_time:.1f}s ({tune_time/60:.1f} min)")
        print(f"   Best CV MAE: {-search.best_score_:.4f}")
        print(f"   Best params: {search.best_params_}")
        
    else:
        print(f"\n🔄 Training baseline model...")
        train_start = time.time()
        
        # Split train into train/validation for early stopping
        val_split = int(len(X_train) * 0.9)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]
        
        print(f"   Using validation set for early stopping ({len(X_val)} samples)")
        
        model = xgb.XGBRegressor(**config)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        train_time = time.time() - train_start
        
        # Get actual number of trees used (after early stopping)
        best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else config['n_estimators']
        
        print(f"✅ Training complete in {train_time:.2f}s")
        print(f"   Speed: {len(X_train) / train_time:.0f} samples/sec")
        print(f"   Trees used: {best_iteration} (early stopping saved {config['n_estimators'] - best_iteration} trees)")
        if device_name == "GPU (CUDA)":
            print(f"   🚀 GPU accelerated training!")
    
    # Evaluate
    print(f"\n📊 Evaluating model...")
    eval_start = time.time()
    y_pred = np.clip(model.predict(X_test), 0, 10)
    eval_time = time.time() - eval_start
    
    metrics = comprehensive_metrics(y_test, y_pred)
    
    print(f"   Inference time: {eval_time:.3f}s ({len(X_test)/eval_time:.0f} samples/sec)")
    print(f"\n� Performance Metrics:")
    print(f"   MAE:           {metrics['mae']:.4f}")
    print(f"   RMSE:          {metrics['rmse']:.4f}")
    print(f"   R²:            {metrics['r2']:.4f}")
    print(f"   Weighted MAE:  {metrics['weighted_mae']:.4f}")
    print(f"   Acc (±1):      {metrics['acc_within_1']:.1%}")
    print(f"   Acc (±2):      {metrics['acc_within_2']:.1%}")
    
    if metrics['high_count'] > 0:
        print(f"   High Days MAE: {metrics['high_mae']:.4f} ({metrics['high_count']} days)")
    
    # Performance assessment
    if metrics['mae'] <= 0.35:
        emoji = "🏆 EXCEPTIONAL"
    elif metrics['mae'] <= 0.45:
        emoji = "✨ EXCELLENT"
    elif metrics['mae'] <= 0.60:
        emoji = "👍 GOOD"
    else:
        emoji = "📊 ACCEPTABLE"
    
    print(f"   {emoji} performance!")
    
    # Feature importance
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n📊 Top 5 Features:")
    for i, (feat, imp) in enumerate(top_features[:5], 1):
        print(f"   {i}. {feat:<25} {imp:.4f}")
    
    return model, metrics, y_pred, feature_importance


# ============================================================================
# MAIN
# ============================================================================

def main():
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train XGBoost models for pollen prediction')
    parser.add_argument('-y', '--yes', action='store_true', 
                        help='Automatically answer yes to all prompts (skip tuning, overwrite models)')
    parser.add_argument('--tune', action='store_true',
                        help='Enable hyperparameter tuning (overrides --yes for tuning choice)')
    args = parser.parse_args()
    
    overall_start = time.time()
    
    print("="*70)
    print("🌸 MULTI-TYPE XGBOOST POLLEN PREDICTOR (GPU-ACCELERATED)")
    print("="*70)
    print("✨ Features:")
    print("   • AAAAI biological thresholds")
    print("   • Separate models for Tree, Grass, Weed, Ragweed, Total")
    print("   • Refined features (weather dynamics + peak proximity)")
    print("   • GPU acceleration (RTX 2080)")
    print("="*70)
    print(f"⏰ Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.yes:
        print("⚡ Auto-mode enabled (--yes flag)")
    if args.tune:
        print("🔍 Hyperparameter tuning enabled (--tune flag)")
    
    # Load data
    from pollen_intensity_predictor import load_and_prepare_data
    
    print(f"\n{'[1/4]':<8} Loading data...")
    load_start = time.time()
    df = load_and_prepare_data()
    load_time = time.time() - load_start
    print(f"         ✅ Loaded {len(df):,} samples in {load_time:.2f}s")
    print(f"         📅 Date range: {df['Date_Standard'].min()} to {df['Date_Standard'].max()}")
    
    # Create biological severity scales
    print(f"\n{'[2/4]':<8} Creating AAAAI-based severity scales...")
    sev_start = time.time()
    df = create_biological_severity_scales(df)
    sev_time = time.time() - sev_start
    print(f"         ✅ Created 5 biological severity scales in {sev_time:.2f}s")
    
    # Engineer refined features
    print(f"\n{'[3/4]':<8} Engineering features...")
    feat_start = time.time()
    df = engineer_refined_features(df)
    feature_cols = select_refined_features(df)
    feat_time = time.time() - feat_start
    print(f"         ✅ Engineered {len(feature_cols)} refined features in {feat_time:.2f}s")
    
    # Train models for each pollen type
    print(f"\n{'[4/4]':<8} Training models...")
    pollen_types = ['Tree', 'Grass', 'Weed', 'Ragweed', 'Total_Pollen']
    results_summary = {}
    training_times = {}
    
    # Determine tuning mode once at the start
    if args.tune:
        do_tuning = True
        print(f"\n🔍 Hyperparameter tuning enabled (--tune flag)")
        print(f"   ⏱️  Each model will take ~4-5 minutes with tuning...")
    elif args.yes:
        do_tuning = False
        print(f"\n⚡ Quick mode enabled (--yes flag, no tuning)")
        print(f"   ⏱️  Each model will take ~30-40 seconds")
    else:
        print(f"\n💡 Tip: Skip tuning for fast results (~2 min total), or tune for best (~20 min)")
        tune_choice = input(f"🤔 Run hyperparameter tuning? [y/n]: ").lower()
        do_tuning = tune_choice == 'y'
        
        if do_tuning:
            print(f"   ⏱️  Each model will take ~4-5 minutes with tuning...")
        else:
            print(f"   ⚡ Quick mode: ~30-40 seconds per model")
    
    models_start = time.time()
    
    for idx, pollen_type in enumerate(pollen_types, 1):
        severity_col = f'{pollen_type}_Severity'
        
        if severity_col not in df.columns:
            print(f"\n⚠️  Skipping {pollen_type} (no data)")
            continue
        
        # Check if model already exists (resume capability)
        model_path = f'pollen_xgboost_{pollen_type.lower()}.joblib'
        import os
        if os.path.exists(model_path) and not args.yes:
            print(f"\n{'='*70}")
            print(f"[Model {idx}/{len(pollen_types)}] {pollen_type}")
            print(f"{'='*70}")
            resume = input(f"⚠️  Model already exists: {model_path}\n   Retrain? [y/n]: ").lower()
            if resume != 'y':
                print(f"   ⏭️  Skipping {pollen_type} (keeping existing model)")
                continue
        elif os.path.exists(model_path) and args.yes:
            print(f"\n⚠️  Overwriting existing model: {model_path} (--yes flag)")
        
        print(f"\n{'='*70}")
        print(f"[Model {idx}/{len(pollen_types)}] {pollen_type}")
        print(f"{'='*70}")
        
        ptype_start = time.time()
        
        # Prepare data
        df_clean = df.dropna(subset=feature_cols + [severity_col])
        split_idx = int(len(df_clean) * 0.8)
        
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df[severity_col]
        X_test = test_df[feature_cols]
        y_test = test_df[severity_col]
        
        print(f"📊 Dataset: {len(X_train):,} train, {len(X_test):,} test")
        print(f"🎯 Target: {severity_col}")
        print(f"   Train range: [{y_train.min():.2f}, {y_train.max():.2f}], mean={y_train.mean():.2f}")
        
        # Train
        model, metrics, y_pred, feature_imp = train_xgboost_for_pollen_type(
            X_train, y_train, X_test, y_test, 
            pollen_type, use_gpu=True, tune=do_tuning
        )
        
        ptype_time = time.time() - ptype_start
        training_times[pollen_type] = ptype_time
        
        # Save model and feature importance
        model_path = f'pollen_xgboost_{pollen_type.lower()}.joblib'
        joblib.dump(model, model_path)
        
        # Convert numpy types to Python native types for JSON serialization
        feature_imp_serializable = {k: float(v) for k, v in feature_imp.items()}
        feature_imp_path = f'xgboost_features_{pollen_type.lower()}.json'
        with open(feature_imp_path, 'w') as f:
            json.dump(feature_imp_serializable, f, indent=2)
        
        print(f"\n💾 Saved: {model_path}")
        print(f"💾 Saved: {feature_imp_path}")
        print(f"⏱️  {pollen_type} total time: {ptype_time:.1f}s ({ptype_time/60:.1f} min)")
        
        results_summary[pollen_type] = metrics
        results_summary[pollen_type]['predictions'] = y_pred.tolist()
        results_summary[pollen_type]['actuals'] = y_test.tolist()
        
        # Progress update
        elapsed = time.time() - models_start
        avg_time = elapsed / idx
        remaining = avg_time * (len(pollen_types) - idx)
        print(f"\n📊 Progress: {idx}/{len(pollen_types)} models complete")
        if idx < len(pollen_types):
            print(f"⏳ Estimated time remaining: {remaining:.0f}s ({remaining/60:.1f} min)")
    
    total_training_time = time.time() - models_start
    
    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Pollen Type':<15} {'MAE':<8} {'R²':<8} {'Acc(±1)':<10} {'Train Time':<12}")
    print("-"*70)
    for ptype, metrics in results_summary.items():
        time_str = f"{training_times[ptype]:.1f}s"
        print(f"{ptype:<15} {metrics['mae']:<8.4f} {metrics['r2']:<8.4f} {metrics['acc_within_1']:<10.1%} {time_str:<12}")
    
    # Averages
    avg_mae = np.mean([m['mae'] for m in results_summary.values()])
    avg_r2 = np.mean([m['r2'] for m in results_summary.values()])
    
    print("-"*70)
    print(f"{'AVERAGE':<15} {avg_mae:<8.4f} {avg_r2:<8.4f}")
    print(f"{'TOTAL TRAINING':<15} {'':<8} {'':<8} {'':<10} {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
    
    # Save summary
    summary_file = {}
    for k, v in results_summary.items():
        summary_file[k] = {}
        for mk, mv in v.items():
            # Handle different types
            if isinstance(mv, (list, np.ndarray)):
                summary_file[k][mk] = [float(x) for x in mv]
            elif isinstance(mv, (np.floating, np.integer)):
                summary_file[k][mk] = float(mv)
            elif isinstance(mv, float) and np.isnan(mv):
                summary_file[k][mk] = None
            else:
                summary_file[k][mk] = mv
    
    # Add timing info
    summary_file['_timing'] = {
        'total_seconds': float(total_training_time),
        'per_model': {k: float(v) for k, v in training_times.items()}
    }
    
    with open('xgboost_multitype_results.json', 'w') as f:
        json.dump(summary_file, f, indent=2)
    
    joblib.dump(feature_cols, 'xgboost_features.joblib')
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    
    # 1. Prediction scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (ptype, metrics) in enumerate(results_summary.items()):
        if '_timing' in ptype:
            continue
        
        ax = axes[idx]
        y_true = np.array(metrics['actuals'])
        y_pred = np.array(metrics['predictions'])
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        ax.plot([0, 10], [0, 10], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Severity')
        ax.set_ylabel('Predicted Severity')
        ax.set_title(f'{ptype}\nMAE: {metrics["mae"]:.3f}, R²: {metrics["r2"]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
    
    # Hide extra subplot
    if len(results_summary) - 1 < len(axes):  # -1 for _timing
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('xgboost_predictions_scatter.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: xgboost_predictions_scatter.png")
    plt.close()
    
    # 2. Feature importance comparison (top 15 features per model)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, ptype in enumerate(pollen_types):
        try:
            with open(f'xgboost_features_{ptype.lower()}.json', 'r') as f:
                feat_imp = json.load(f)
            
            top_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
            features_names = [f[0] for f in top_feats]
            importances = [f[1] for f in top_feats]
            
            ax = axes[idx]
            y_pos = np.arange(len(features_names))
            ax.barh(y_pos, importances, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features_names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'{ptype} Top 15 Features')
            ax.grid(True, alpha=0.3, axis='x')
        except Exception as e:
            print(f"   ⚠️  Could not plot {ptype}: {e}")
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance_detailed.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: xgboost_feature_importance_detailed.png")
    plt.close()
    
    overall_time = time.time() - overall_start
    
    print(f"\n💾 Outputs:")
    print(f"   • Models: pollen_xgboost_[type].joblib (5 files)")
    print(f"   • Feature importance: xgboost_features_[type].json (5 files)")
    print(f"   • Results: xgboost_multitype_results.json")
    print(f"   • Features list: xgboost_features.joblib")
    print(f"   • Visualizations: xgboost_predictions_scatter.png")
    print(f"   • Visualizations: xgboost_feature_importance_detailed.png")
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"⏱️  Total runtime: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print(f"⏰ Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🎯 Key Results:")
    print(f"   • Average MAE: {avg_mae:.4f}")
    print(f"   • Average R²: {avg_r2:.4f}")
    print(f"   • {len([k for k in results_summary.keys() if k != '_timing'])} models trained successfully")
    
    # Compare with baseline if available
    try:
        with open('model_results.json', 'r') as f:
            baseline = json.load(f)
            baseline_mae = baseline.get('mae', None)
            if baseline_mae:
                improvement = ((baseline_mae - avg_mae) / baseline_mae) * 100
                print(f"\n📈 Improvement over Random Forest baseline:")
                print(f"   • Baseline MAE: {baseline_mae:.4f}")
                print(f"   • XGBoost MAE: {avg_mae:.4f}")
                print(f"   • Improvement: {improvement:.1f}%")
                if improvement >= 20:
                    print(f"   🎉 Target achieved! (>20% improvement)")
                elif improvement >= 10:
                    print(f"   👍 Good improvement!")
                else:
                    print(f"   📊 Modest improvement")
    except Exception as e:
        print(f"\n   (No baseline comparison available)")
    
    print(f"\n💡 Usage tips:")
    print(f"   • Fast training: python train_xgboost_multitype.py --yes")
    print(f"   • With tuning:   python train_xgboost_multitype.py --tune")
    print(f"   • Both:          python train_xgboost_multitype.py --yes --tune")
    
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("   Already-saved models are preserved.")
        print("   Re-run the script to resume from where you left off.\n")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        print(f"\n📋 Traceback:")
        traceback.print_exc()
        exit(1)
