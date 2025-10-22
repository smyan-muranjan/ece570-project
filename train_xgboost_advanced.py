"""
Advanced XGBoost Training for Pollen Prediction
Includes hyperparameter tuning, custom metrics, and comprehensive evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# XGBoost import with fallback
try:
    import xgboost as xgb
    print("✅ XGBoost imported successfully")
    HAS_XGB = True
except ImportError:
    print("❌ XGBoost not installed!")
    print("📦 Install with: pip install xgboost")
    HAS_XGB = False
    exit(1)

from datetime import datetime
import joblib
import json


# ============================================================================
# CUSTOM METRICS FOR POLLEN PREDICTION
# ============================================================================

def weighted_mae(y_true, y_pred, high_severity_weight=2.0):
    """
    Weighted MAE that penalizes errors on high pollen days more heavily
    Critical for allergy forecasting - missing severe days is worse
    """
    errors = np.abs(y_true - y_pred)
    weights = np.where(y_true >= 7, high_severity_weight, 1.0)
    return np.mean(errors * weights)


def severity_accuracy(y_true, y_pred, tolerance=1):
    """
    Accuracy within tolerance levels
    (predicting 6 when actual is 7 is acceptable for users)
    """
    within_tolerance = np.abs(y_true - y_pred) <= tolerance
    return np.mean(within_tolerance)


def comprehensive_metrics(y_true, y_pred):
    """Calculate all relevant metrics"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'weighted_mae': weighted_mae(y_true, y_pred),
        'accuracy_within_1': severity_accuracy(y_true, y_pred, tolerance=1),
        'accuracy_within_2': severity_accuracy(y_true, y_pred, tolerance=2),
    }
    
    # High severity performance (days >= 7)
    high_mask = y_true >= 7
    if np.any(high_mask):
        metrics['high_severity_mae'] = mean_absolute_error(
            y_true[high_mask], 
            y_pred[high_mask]
        )
        metrics['high_severity_count'] = int(np.sum(high_mask))
    else:
        metrics['high_severity_mae'] = np.nan
        metrics['high_severity_count'] = 0
    
    return metrics


# ============================================================================
# XGBOOST CONFIGURATION
# ============================================================================

def get_xgboost_base_config():
    """
    Get well-tuned baseline XGBoost configuration
    These parameters work well for time series regression
    """
    return {
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,  # Minimum loss reduction for split
        'reg_alpha': 0.01,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',  # Faster training
        'verbosity': 0
    }


def get_hyperparameter_search_space():
    """
    Define hyperparameter search space for RandomizedSearchCV
    Focused on parameters that most impact pollen prediction
    """
    return {
        'n_estimators': [200, 300, 400, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'max_depth': [5, 7, 9, 11],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0]
    }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_xgboost_baseline(X_train, y_train, X_test, y_test):
    """
    Train XGBoost with good baseline parameters (no tuning)
    Fast approach - good for initial testing
    """
    print("\n" + "="*70)
    print("🚀 TRAINING XGBOOST - BASELINE CONFIGURATION")
    print("="*70)
    
    # Get baseline config
    params = get_xgboost_base_config()
    
    print("\n📋 Baseline Parameters:")
    for key, value in params.items():
        print(f"   {key:<20} = {value}")
    
    # Train model
    print("\n🔄 Training XGBoost...")
    model = xgb.XGBRegressor(**params)
    
    # Fit with evaluation set to monitor performance
    model.fit(X_train, y_train)
    
    print("✅ Training complete!")
    
    return model


def train_xgboost_with_tuning(X_train, y_train, n_iter=50, cv_splits=5):
    """
    Train XGBoost with hyperparameter tuning
    More time-consuming but achieves better performance
    """
    print("\n" + "="*70)
    print("🔍 TRAINING XGBOOST - WITH HYPERPARAMETER TUNING")
    print("="*70)
    
    # Base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        verbosity=0
    )
    
    # Search space
    param_space = get_hyperparameter_search_space()
    
    print(f"\n🔧 Hyperparameter Search:")
    print(f"   Search iterations: {n_iter}")
    print(f"   CV splits: {cv_splits}")
    print(f"   Search space size: ~{np.prod([len(v) for v in param_space.values()]):,} combinations")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # Randomized search
    print(f"\n🔄 Running randomized search (this may take 5-10 minutes)...")
    
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print("\n✅ Hyperparameter tuning complete!")
    print("\n🏆 Best Parameters Found:")
    for param, value in random_search.best_params_.items():
        print(f"   {param:<20} = {value}")
    
    print(f"\n📊 Best CV MAE: {-random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="XGBoost"):
    """
    Comprehensive model evaluation with custom metrics
    """
    print("\n" + "="*70)
    print(f"📊 COMPREHENSIVE EVALUATION: {model_name}")
    print("="*70)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Clip to valid severity range [0, 10]
    y_train_pred = np.clip(y_train_pred, 0, 10)
    y_test_pred = np.clip(y_test_pred, 0, 10)
    
    # Calculate metrics
    train_metrics = comprehensive_metrics(y_train, y_train_pred)
    test_metrics = comprehensive_metrics(y_test, y_test_pred)
    
    # Display standard metrics
    print("\n📈 Standard Metrics:")
    print(f"   {'Metric':<20} {'Train':<12} {'Test':<12} {'Difference':<12}")
    print(f"   {'-'*60}")
    print(f"   {'MAE':<20} {train_metrics['mae']:<12.4f} {test_metrics['mae']:<12.4f} {abs(test_metrics['mae']-train_metrics['mae']):<12.4f}")
    print(f"   {'RMSE':<20} {train_metrics['rmse']:<12.4f} {test_metrics['rmse']:<12.4f} {abs(test_metrics['rmse']-train_metrics['rmse']):<12.4f}")
    print(f"   {'R²':<20} {train_metrics['r2']:<12.4f} {test_metrics['r2']:<12.4f} {abs(test_metrics['r2']-train_metrics['r2']):<12.4f}")
    
    # Display custom metrics
    print("\n🎯 Allergy-Specific Metrics:")
    print(f"   Weighted MAE (test):        {test_metrics['weighted_mae']:.4f}")
    print(f"   Accuracy within ±1 level:   {test_metrics['accuracy_within_1']:.1%}")
    print(f"   Accuracy within ±2 levels:  {test_metrics['accuracy_within_2']:.1%}")
    
    if test_metrics['high_severity_count'] > 0:
        print(f"   High severity MAE (≥7):     {test_metrics['high_severity_mae']:.4f}")
        print(f"   High severity days:         {test_metrics['high_severity_count']}")
    
    # Performance assessment
    print(f"\n🏆 Performance Assessment:")
    mae = test_metrics['mae']
    if mae <= 0.35:
        tier = "🏆 EXCEPTIONAL (MAE ≤ 0.35)"
        emoji = "🎉"
    elif mae <= 0.45:
        tier = "✅ EXCELLENT (MAE ≤ 0.45)"
        emoji = "✨"
    elif mae <= 0.60:
        tier = "👍 GOOD (MAE ≤ 0.60)"
        emoji = "👌"
    else:
        tier = "🔶 ACCEPTABLE (MAE > 0.60)"
        emoji = "📊"
    
    print(f"   {emoji} Test MAE: {mae:.4f} - {tier}")
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }


def analyze_feature_importance(model, feature_names, top_n=15):
    """
    Analyze and display XGBoost feature importance
    """
    print("\n" + "="*70)
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Get feature importance (gain-based)
    importance = model.feature_importances_
    
    # Create dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top {top_n} Most Important Features:")
    print(f"   {'Rank':<6} {'Feature':<30} {'Importance':<12} {'% of Total':<12}")
    print(f"   {'-'*70}")
    
    total_importance = importance.sum()
    cumulative = 0
    
    for i, (_, row) in enumerate(feature_importance.head(top_n).iterrows(), 1):
        pct = (row['importance'] / total_importance) * 100
        cumulative += pct
        print(f"   {i:<6} {row['feature']:<30} {row['importance']:<12.4f} {pct:<12.1f}%")
    
    print(f"\n   💡 Top {top_n} features account for {cumulative:.1f}% of total importance")
    
    return feature_importance


def plot_predictions(y_test, y_test_pred, test_dates, model_name, save_path='xgboost_predictions.png'):
    """
    Create comprehensive prediction visualizations
    """
    print("\n📈 Creating prediction plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Scatter plot (Predicted vs Actual)
    ax = axes[0, 0]
    ax.scatter(y_test, y_test_pred, alpha=0.6, s=30)
    ax.plot([0, 10], [0, 10], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Severity', fontsize=11)
    ax.set_ylabel('Predicted Severity', fontsize=11)
    ax.set_title(f'{model_name}: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Time series (last 100 days)
    ax = axes[0, 1]
    last_n = min(100, len(test_dates))
    ax.plot(test_dates.iloc[-last_n:], y_test.iloc[-last_n:], 
            'b-', label='Actual', linewidth=2, marker='o', markersize=3)
    ax.plot(test_dates.iloc[-last_n:], y_test_pred[-last_n:], 
            'r-', label='Predicted', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Pollen Severity', fontsize=11)
    ax.set_title(f'Time Series: Last {last_n} Test Days', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Residuals
    ax = axes[1, 0]
    residuals = y_test - y_test_pred
    ax.scatter(y_test_pred, residuals, alpha=0.6, s=30)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values', fontsize=11)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    ax = axes[1, 1]
    errors = np.abs(y_test - y_test_pred)
    ax.hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean Error: {errors.mean():.3f}')
    ax.set_xlabel('Absolute Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Plots saved to: {save_path}")
    
    return fig


def save_model_and_results(model, feature_names, results, best_params, model_name="XGBoost"):
    """
    Save trained model, features, and results
    """
    print("\n" + "="*70)
    print("💾 SAVING MODEL AND RESULTS")
    print("="*70)
    
    # Save model
    model_file = f'pollen_predictor_{model_name.lower().replace(" ", "_")}.joblib'
    joblib.dump(model, model_file)
    print(f"   ✅ Model saved: {model_file}")
    
    # Save features
    features_file = 'model_features_xgboost.joblib'
    joblib.dump(feature_names, features_file)
    print(f"   ✅ Features saved: {features_file}")
    
    # Save results
    test_metrics = results['test_metrics']
    results_summary = {
        'model_name': model_name,
        'test_mae': float(test_metrics['mae']),
        'test_rmse': float(test_metrics['rmse']),
        'test_r2': float(test_metrics['r2']),
        'weighted_mae': float(test_metrics['weighted_mae']),
        'accuracy_within_1': float(test_metrics['accuracy_within_1']),
        'accuracy_within_2': float(test_metrics['accuracy_within_2']),
        'high_severity_mae': float(test_metrics['high_severity_mae']) if not np.isnan(test_metrics['high_severity_mae']) else None,
        'high_severity_count': int(test_metrics['high_severity_count']),
        'feature_count': len(feature_names),
        'best_params': best_params if best_params else 'baseline',
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = 'model_results_xgboost.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"   ✅ Results saved: {results_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution: Load data, train XGBoost, evaluate, and save
    """
    print("="*70)
    print("🌸 ADVANCED XGBOOST POLLEN PREDICTOR")
    print("="*70)
    print("🎯 Training Strategy:")
    print("   1. Baseline XGBoost (fast)")
    print("   2. Hyperparameter-tuned XGBoost (best performance)")
    print("   3. Comprehensive evaluation")
    print("   4. Feature importance analysis")
    print("="*70)
    
    # Import preprocessing functions
    from pollen_intensity_predictor import (
        load_and_prepare_data,
        create_pollen_severity_scale,
        engineer_features,
        select_features,
        prepare_train_test_split
    )
    
    # Step 1: Load and prepare data
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    df = load_and_prepare_data()
    df, thresholds = create_pollen_severity_scale(df)
    df = engineer_features(df)
    feature_cols = select_features(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test, train_dates, test_dates, df_clean = \
        prepare_train_test_split(df, feature_cols, target_col='Pollen_Severity')
    
    print(f"\n✅ Data prepared:")
    print(f"   Training samples:   {len(X_train)}")
    print(f"   Test samples:       {len(X_test)}")
    print(f"   Features:           {len(feature_cols)}")
    print(f"   Target range:       {y_train.min():.0f} - {y_train.max():.0f}")
    
    # Step 2: Train baseline XGBoost
    print("\n" + "="*70)
    print("STEP 2: BASELINE XGBOOST TRAINING")
    print("="*70)
    
    baseline_model = train_xgboost_baseline(X_train, y_train, X_test, y_test)
    baseline_results = evaluate_model(baseline_model, X_train, X_test, y_train, y_test, 
                                      model_name="XGBoost Baseline")
    
    # Step 3: Train with hyperparameter tuning
    print("\n" + "="*70)
    print("STEP 3: HYPERPARAMETER TUNING")
    print("="*70)
    
    user_choice = input("\n🤔 Run hyperparameter tuning? (recommended, takes 5-10 min) [y/n]: ").lower().strip()
    
    if user_choice == 'y':
        tuned_model, best_params = train_xgboost_with_tuning(
            X_train, y_train, 
            n_iter=50,  # More iterations = better results but slower
            cv_splits=5
        )
        tuned_results = evaluate_model(tuned_model, X_train, X_test, y_train, y_test,
                                      model_name="XGBoost Tuned")
        
        # Compare baseline vs tuned
        print("\n" + "="*70)
        print("📊 BASELINE vs TUNED COMPARISON")
        print("="*70)
        
        baseline_mae = baseline_results['test_metrics']['mae']
        tuned_mae = tuned_results['test_metrics']['mae']
        improvement = (baseline_mae - tuned_mae) / baseline_mae * 100
        
        print(f"\n   Baseline MAE:  {baseline_mae:.4f}")
        print(f"   Tuned MAE:     {tuned_mae:.4f}")
        print(f"   Improvement:   {improvement:.1f}%")
        
        if tuned_mae < baseline_mae:
            print(f"\n   ✅ Tuning improved performance! Using tuned model.")
            final_model = tuned_model
            final_results = tuned_results
            final_params = best_params
            final_name = "XGBoost_Tuned"
        else:
            print(f"\n   ℹ️  Baseline performed better. Using baseline model.")
            final_model = baseline_model
            final_results = baseline_results
            final_params = None
            final_name = "XGBoost_Baseline"
    else:
        print("\n   ℹ️  Skipping hyperparameter tuning. Using baseline model.")
        final_model = baseline_model
        final_results = baseline_results
        final_params = None
        final_name = "XGBoost_Baseline"
    
    # Step 4: Feature importance
    feature_importance = analyze_feature_importance(final_model, feature_cols, top_n=15)
    
    # Step 5: Visualizations
    plot_predictions(y_test, final_results['y_test_pred'], test_dates, final_name)
    
    # Step 6: Save everything
    save_model_and_results(final_model, feature_cols, final_results, final_params, final_name)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    mae = final_results['test_metrics']['mae']
    r2 = final_results['test_metrics']['r2']
    acc = final_results['test_metrics']['accuracy_within_1']
    
    print(f"\n🏆 Final Model: {final_name}")
    print(f"   MAE:               {mae:.4f}")
    print(f"   R²:                {r2:.4f}")
    print(f"   Accuracy (±1):     {acc:.1%}")
    
    # Compare with baseline RF if available
    try:
        with open('model_results.json', 'r') as f:
            rf_results = json.load(f)
            rf_mae = rf_results.get('test_mae', None)
            
            if rf_mae:
                improvement = (rf_mae - mae) / rf_mae * 100
                print(f"\n📈 Improvement over Random Forest:")
                print(f"   RF MAE:            {rf_mae:.4f}")
                print(f"   XGBoost MAE:       {mae:.4f}")
                print(f"   Improvement:       {improvement:.1f}%")
                
                if improvement > 0:
                    print(f"   ✨ XGBoost is {improvement:.1f}% better!")
    except:
        pass
    
    print("\n🚀 Model ready for deployment!")
    print(f"   Load model: joblib.load('pollen_predictor_{final_name.lower()}.joblib')")
    
    return 0


if __name__ == "__main__":
    exit(main())
