"""
Optimized XGBoost for Total_Pollen Only
Direct comparison with Random Forest baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import joblib
import json
import time
import argparse


# ============================================================================
# CONFIGURATION
# ============================================================================

def get_optimized_xgboost_config():
    """Optimized XGBoost config for Total_Pollen (addressing RF comparison)"""
    return {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',  # GPU acceleration
        # Adjusted hyperparameters to beat Random Forest
        'n_estimators': 1000,  # More trees with early stopping
        'learning_rate': 0.03,  # Lower learning rate for better generalization
        'max_depth': 6,  # Slightly shallower than RF (prevents overfitting)
        'min_child_weight': 5,  # Higher for more conservative splits
        'subsample': 0.85,  # Slight increase for better generalization
        'colsample_bytree': 0.85,
        'colsample_bylevel': 0.7,  # Additional regularization
        'gamma': 0.05,  # Reduced for more splits
        'reg_alpha': 0.05,  # L1 regularization
        'reg_lambda': 1.5,  # Stronger L2 regularization
        'random_state': 42,
        'verbosity': 0
    }


def get_tuning_space():
    """Hyperparameter search space for aggressive tuning"""
    return {
        'n_estimators': [500, 800, 1000, 1200],
        'learning_rate': [0.01, 0.02, 0.03, 0.05],
        'max_depth': [5, 6, 7, 8],
        'min_child_weight': [3, 5, 7, 10],
        'subsample': [0.75, 0.8, 0.85, 0.9],
        'colsample_bytree': [0.75, 0.8, 0.85, 0.9],
        'colsample_bylevel': [0.6, 0.7, 0.8],
        'gamma': [0, 0.05, 0.1, 0.15],
        'reg_alpha': [0, 0.01, 0.05, 0.1],
        'reg_lambda': [1.0, 1.5, 2.0, 2.5]
    }


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    """Load and prepare data using exact baseline pipeline"""
    from rf import (
        load_and_prepare_data, 
        create_pollen_severity_scale,
        engineer_features
    )
    print("📂 Loading data...")
    df = load_and_prepare_data()
    df, thresholds = create_pollen_severity_scale(df)
    df = engineer_features(df)
    print(f"   ✅ Loaded {len(df):,} samples")
    print(f"   📅 {df['Date_Standard'].min()} to {df['Date_Standard'].max()}")
    return df


# ============================================================================
# FEATURE SELECTION (using baseline's exact feature selection)
# ============================================================================

def select_features(df):
    """Use exact same feature selection as Random Forest baseline"""
    from rf import select_features as baseline_select
    return baseline_select(df)


# ============================================================================
# TRAINING
# ============================================================================

def train_xgboost(X_train, y_train, X_test, y_test, tune=False):
    """Train XGBoost with early stopping"""
    print("\n" + "="*70)
    print("🚀 TRAINING XGBOOST FOR TOTAL_POLLEN")
    print("="*70)
    
    # Check GPU
    print("\n🔍 Checking GPU...")
    try:
        test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
        test_model.fit(X_train[:100], y_train[:100])
        print("✅ RTX 2080 GPU detected and working!")
        config = get_optimized_xgboost_config()
    except Exception as e:
        print(f"⚠️  GPU not available, using CPU")
        config = get_optimized_xgboost_config()
        config['device'] = 'cpu'
    
    # Split for early stopping (use 15% validation instead of 10%)
    val_split = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_tr):,} samples")
    print(f"   Validation: {len(X_val):,} samples (for early stopping)")
    print(f"   Test: {len(X_test):,} samples")
    
    if tune:
        print(f"\n🔍 Hyperparameter tuning (50 iterations)...")
        train_start = time.time()
        
        base = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method=config['tree_method'],
            device=config['device'],
            random_state=42,
            verbosity=0
        )
        
        search = RandomizedSearchCV(
            base, get_tuning_space(),
            n_iter=50,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=1,
            verbose=2
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        train_time = time.time() - train_start
        print(f"\n✅ Tuning complete in {train_time:.1f}s ({train_time/60:.1f} min)")
        print(f"   Best CV MAE: {-search.best_score_:.4f}")
        print(f"   Best params: {search.best_params_}")
        
    else:
        print(f"\n🔄 Training with optimized config...")
        print(f"   Trees: {config['n_estimators']}")
        print(f"   Learning rate: {config['learning_rate']}")
        print(f"   Max depth: {config['max_depth']}")
        print(f"   Early stopping: after 100 rounds without improvement")
        
        train_start = time.time()
        
        # In XGBoost 3.1+, early_stopping_rounds goes in constructor
        config_with_early_stop = config.copy()
        config_with_early_stop['early_stopping_rounds'] = 100
        
        model = xgb.XGBRegressor(**config_with_early_stop)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        train_time = time.time() - train_start
        
        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else config['n_estimators']
        
        print(f"\n✅ Training complete in {train_time:.2f}s")
        print(f"   Speed: {len(X_tr) / train_time:.0f} samples/sec")
        print(f"   Trees used: {best_iter} / {config['n_estimators']}")
        print(f"   Early stopping saved: {config['n_estimators'] - best_iter} trees")
        if config['device'] == 'cuda':
            print(f"   🚀 GPU accelerated!")
    
    # Evaluate
    print(f"\n📊 Evaluating...")
    y_pred = np.clip(model.predict(X_test), 0, 10)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Additional metrics
    acc_within_1 = np.mean(np.abs(y_test - y_pred) <= 1)
    acc_within_2 = np.mean(np.abs(y_test - y_pred) <= 2)
    
    # High severity days (>=7)
    high_mask = y_test >= 7
    high_mae = mean_absolute_error(y_test[high_mask], y_pred[high_mask]) if np.any(high_mask) else np.nan
    
    print(f"\n{'='*70}")
    print("📈 PERFORMANCE METRICS")
    print(f"{'='*70}")
    print(f"MAE:           {mae:.4f}")
    print(f"RMSE:          {rmse:.4f}")
    print(f"R²:            {r2:.4f}")
    print(f"Acc (±1):      {acc_within_1:.1%}")
    print(f"Acc (±2):      {acc_within_2:.1%}")
    if not np.isnan(high_mae):
        print(f"High Days MAE: {high_mae:.4f} ({int(np.sum(high_mask))} days)")
    
    # Compare with Random Forest
    print(f"\n{'='*70}")
    print("📊 COMPARISON WITH RANDOM FOREST BASELINE")
    print(f"{'='*70}")
    
    try:
        with open('results/rf.json', 'r') as f:
            baseline = json.load(f)
            rf_mae = baseline['test_mae']
            rf_r2 = baseline['test_r2']
            
            improvement = ((rf_mae - mae) / rf_mae) * 100
            r2_change = ((r2 - rf_r2) / rf_r2) * 100
            
            print(f"\n{'Metric':<15} {'Random Forest':<15} {'XGBoost':<15} {'Change':<15}")
            print("-"*70)
            print(f"{'MAE':<15} {rf_mae:<15.4f} {mae:<15.4f} {improvement:>+.1f}%")
            print(f"{'R²':<15} {rf_r2:<15.4f} {r2:<15.4f} {r2_change:>+.1f}%")
            
            if improvement > 0:
                print(f"\n🎉 XGBoost is {improvement:.1f}% better!")
                if improvement >= 20:
                    print(f"   ✅ TARGET ACHIEVED! (>20% improvement)")
                elif improvement >= 10:
                    print(f"   👍 Good improvement!")
            else:
                print(f"\n⚠️  XGBoost is {abs(improvement):.1f}% worse than Random Forest")
                print(f"   Consider running with --tune flag for better results")
    except Exception as e:
        print(f"(No baseline comparison available)")
    
    results = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'acc_within_1': float(acc_within_1),
        'acc_within_2': float(acc_within_2),
        'high_mae': float(high_mae) if not np.isnan(high_mae) else None,
        'high_count': int(np.sum(high_mask)),
        'train_time': train_time,
        'best_iteration': int(best_iter) if not tune else None
    }
    
    return model, results, y_pred, y_test


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train optimized XGBoost for Total_Pollen')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning (50 iterations)')
    args = parser.parse_args()
    
    overall_start = time.time()
    
    print("="*70)
    print("🌸 XGBOOST TOTAL_POLLEN OPTIMIZER")
    print("="*70)
    print("🎯 Goal: Beat Random Forest baseline (MAE: 0.486)")
    if args.tune:
        print("🔍 Mode: Hyperparameter tuning enabled (~15-20 min)")
    else:
        print("⚡ Mode: Optimized config (~30-40 seconds)")
    print("="*70)
    
    # Load data (already has features and Pollen_Severity from baseline)
    df = load_data()
    features = select_features(df)
    
    # Use exact same target as Random Forest baseline
    target = 'Pollen_Severity'
    
    # Split data (80/20 time series split)
    df_clean = df.dropna(subset=features + [target])
    split_idx = int(len(df_clean) * 0.8)
    
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    print(f"\n📅 Train: {train_df['Date_Standard'].min()} to {train_df['Date_Standard'].max()}")
    print(f"📅 Test:  {test_df['Date_Standard'].min()} to {test_df['Date_Standard'].max()}")
    
    # Train
    model, results, y_pred, y_true = train_xgboost(X_train, y_train, X_test, y_test, tune=args.tune)
    
    # Save
    print(f"\n💾 Saving outputs...")
    joblib.dump(model, 'models/xg_boost_baseline.joblib')
    print(f"   ✅ Model: pollen_xgboost_total_pollen_optimized.joblib")
    
    # Save feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    with open('results/xgboost.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✅ Results: xgboost_total_pollen_results.json")
    
    # Create combined plot with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Feature importance plot
    names = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]
    
    ax1.barh(range(len(names)), importances, color='steelblue')
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Importance', fontsize=11)
    ax1.set_title('XGBoost Total_Pollen: Top 20 Features', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Right: Prediction scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5, s=30, color='steelblue')
    ax2.plot([0, 10], [0, 10], 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Severity (0-10)', fontsize=11)
    ax2.set_ylabel('Predicted Severity (0-10)', fontsize=11)
    ax2.set_title(f'Predictions\nMAE: {results["mae"]:.3f}, R²: {results["r2"]:.3f}', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('results/xgboost.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ Plot: results/xgboost.png")
    
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"⏱️  Total runtime: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print(f"\n💡 Next steps:")
    if not args.tune and results['mae'] > 0.45:
        print(f"   • Try: python train_xgboost_total_pollen.py --tune")
        print(f"   • This will search for better hyperparameters")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
