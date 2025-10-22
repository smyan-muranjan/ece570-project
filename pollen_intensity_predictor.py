"""
Pollen Intensity Predictor - Baseline Model
Predicts daily pollen levels using weather and temporal features
Foundation model for allergy forecasting system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
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
    
    # Use Total_Pollen as our target
    pollen_data = df['Total_Pollen'].copy()
    
    # Define severity scale based on percentiles
    percentiles = [0, 10, 25, 40, 55, 70, 80, 85, 90, 95, 100]
    thresholds = np.percentile(pollen_data[pollen_data > 0], percentiles)
    
    print("   Pollen Severity Scale:")
    severity_labels = ['None', 'Very Low', 'Low', 'Low-Mod', 'Moderate', 
                      'Mod-High', 'High', 'Very High', 'Extreme', 'Severe']
    
    for i, (label, threshold) in enumerate(zip(severity_labels, thresholds)):
        print(f"   {i}: {label:10} >= {threshold:6.1f} pollen count")
    
    # Create severity score (0-10)
    def pollen_to_severity(pollen_count):
        if pollen_count == 0:
            return 0
        for i in range(len(thresholds)-1, 0, -1):
            if pollen_count >= thresholds[i]:
                return i
        return 0
    
    df['Pollen_Severity'] = df['Total_Pollen'].apply(pollen_to_severity)
    
    print(f"\n   Severity distribution:")
    severity_dist = df['Pollen_Severity'].value_counts().sort_index()
    for severity, count in severity_dist.items():
        label = severity_labels[severity] if severity < len(severity_labels) else f"Level {severity}"
        print(f"   {severity}: {label:10} - {count:4d} days ({count/len(df)*100:5.1f}%)")
    
    return df, thresholds

def engineer_features(df):
    """Engineer temporal and weather features for prediction"""
    print("\n🔧 Engineering features...")
     
    # Sort by date for time-based features
    df = df.sort_values('Date_Standard').reset_index(drop=True)
    
    # Temporal features
    df['Year'] = df['Date_Standard'].dt.year
    df['Month'] = df['Date_Standard'].dt.month
    df['Day_of_Year'] = df['Date_Standard'].dt.dayofyear
    df['Day_of_Week'] = df['Date_Standard'].dt.dayofweek
    
    # Seasonal features (cyclical encoding)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_of_Year_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
    df['Day_of_Year_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
    
    # Weather features
    df['Temp_Range'] = df['TMAX'] - df['TMIN']
    df['Is_Rainy'] = (df['PRCP'] > 0).astype(int)
    df['High_Wind'] = (df['AWND'] > df['AWND'].quantile(0.75)).astype(int)
    
    # Lag features (previous days' pollen and weather)
    for lag in [1, 3, 7]:
        df[f'Pollen_lag_{lag}'] = df['Total_Pollen'].shift(lag)
        df[f'TMAX_lag_{lag}'] = df['TMAX'].shift(lag)
        df[f'PRCP_lag_{lag}'] = df['PRCP'].shift(lag)
    
    # Rolling averages
    for window in [3, 7, 14]:
        df[f'Pollen_roll_mean_{window}'] = df['Total_Pollen'].rolling(window=window, min_periods=1).mean()
        df[f'Temp_roll_mean_{window}'] = df['TAVG'].rolling(window=window, min_periods=1).mean()
        df[f'Precip_roll_sum_{window}'] = df['PRCP'].rolling(window=window, min_periods=1).sum()
    
    # Growing degree days (cumulative temperature for plant growth)
    base_temp = 5  # Base temperature for pollen production
    df['GDD'] = np.maximum(df['TAVG'] - base_temp, 0)
    df['GDD_cumsum'] = df.groupby(df['Date_Standard'].dt.year)['GDD'].cumsum()
    
    print(f"   Created {len([col for col in df.columns if any(x in col for x in ['lag_', 'roll_', '_sin', '_cos', 'GDD', 'Temp_Range'])])} engineered features")
    
    return df

def select_features(df):
    """Select features for model training"""
    print("\n📋 Selecting features for training...")
    
    # Core weather features
    weather_features = ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND', 'Temp_Range', 'Is_Rainy', 'High_Wind']
    
    # Temporal features
    temporal_features = ['Month', 'Day_of_Year', 'Day_of_Week', 
                        'Month_sin', 'Month_cos', 'Day_of_Year_sin', 'Day_of_Year_cos']
    
    # Lag features
    lag_features = [col for col in df.columns if 'lag_' in col]
    
    # Rolling features
    rolling_features = [col for col in df.columns if 'roll_' in col]
    
    # Growing degree days
    gdd_features = ['GDD', 'GDD_cumsum']
    
    # Combine all features
    feature_cols = weather_features + temporal_features + lag_features + rolling_features + gdd_features
    
    # Filter to available columns
    available_features = [col for col in feature_cols if col in df.columns]
    
    print(f"   Selected {len(available_features)} features:")
    print(f"   Weather: {len([f for f in available_features if f in weather_features])}")
    print(f"   Temporal: {len([f for f in available_features if f in temporal_features])}")
    print(f"   Lag: {len([f for f in available_features if 'lag_' in f])}")
    print(f"   Rolling: {len([f for f in available_features if 'roll_' in f])}")
    print(f"   GDD: {len([f for f in available_features if f in gdd_features])}")
    
    return available_features

def prepare_train_test_split(df, feature_cols, target_col='Pollen_Severity'):
    """Prepare train/test split with temporal considerations"""
    print(f"\n📊 Preparing train/test split...")
    
    # Remove rows with missing lag features (first few days)
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    
    print(f"   Clean dataset: {len(df_clean)} samples")
    print(f"   Date range: {df_clean['Date_Standard'].min().date()} to {df_clean['Date_Standard'].max().date()}")
    
    # Time-based split (80% train, 20% test)
    # Use chronological split to simulate real-world prediction
    split_date = df_clean['Date_Standard'].quantile(0.8)
    
    train_mask = df_clean['Date_Standard'] <= split_date
    test_mask = df_clean['Date_Standard'] > split_date
    
    X_train = df_clean.loc[train_mask, feature_cols]
    X_test = df_clean.loc[test_mask, feature_cols]
    y_train = df_clean.loc[train_mask, target_col]
    y_test = df_clean.loc[test_mask, target_col]
    
    # Also get dates for analysis
    train_dates = df_clean.loc[train_mask, 'Date_Standard']
    test_dates = df_clean.loc[test_mask, 'Date_Standard']
    
    print(f"   Train set: {len(X_train)} samples ({train_dates.min().date()} to {train_dates.max().date()})")
    print(f"   Test set:  {len(X_test)} samples ({test_dates.min().date()} to {test_dates.max().date()})")
    print(f"   Split date: {split_date.date()}")
    
    return X_train, X_test, y_train, y_test, train_dates, test_dates, df_clean

def train_models(X_train, y_train):
    """Train multiple models and compare performance"""
    print("\n🤖 Training models...")
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Ridge Regression': Ridge(alpha=1.0),
        'Linear Regression': LinearRegression()
    }
    
    trained_models = {}
    cv_scores = {}
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Cross-validation
        cv_score = cross_val_score(model, X_train, y_train, cv=tscv, 
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_scores[name] = -cv_score.mean()
        
        print(f"     CV MAE: {cv_scores[name]:.3f} ± {cv_score.std():.3f}")
    
    # Select best model
    best_model_name = min(cv_scores.keys(), key=lambda k: cv_scores[k])
    best_model = trained_models[best_model_name]
    
    print(f"\n   🏆 Best model: {best_model_name} (CV MAE: {cv_scores[best_model_name]:.3f})")
    
    return trained_models, best_model, best_model_name

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation"""
    print(f"\n📊 Evaluating {model_name}...")
    
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
    
    print(f"   📈 Performance Metrics:")
    print(f"   {'Metric':<12} {'Train':<8} {'Test':<8} {'Difference':<10}")
    print(f"   {'-'*40}")
    print(f"   {'MAE':<12} {train_mae:<8.3f} {test_mae:<8.3f} {abs(test_mae-train_mae):<10.3f}")
    print(f"   {'RMSE':<12} {train_rmse:<8.3f} {test_rmse:<8.3f} {abs(test_rmse-train_rmse):<10.3f}")
    print(f"   {'R²':<12} {train_r2:<8.3f} {test_r2:<8.3f} {abs(test_r2-train_r2):<10.3f}")
    
    # Performance tiers (from project goals)
    print(f"\n   🎯 Performance Assessment (MAE on 0-10 scale):")
    if test_mae <= 1.0:
        tier = "🏆 EXCELLENT"
    elif test_mae <= 2.0:
        tier = "✅ GOOD"
    elif test_mae <= 3.0:
        tier = "🔶 OKAY"
    else:
        tier = "❌ POOR"
    
    print(f"   Test MAE: {test_mae:.3f} - {tier}")
    
    return {
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_pred': y_train_pred, 'test_pred': y_test_pred
    }

def analyze_feature_importance(model, feature_cols, model_name):
    """Analyze feature importance"""
    print(f"\n🔍 Feature Importance Analysis ({model_name})...")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:<20} {row['importance']:.4f}")
        
        return feature_importance
    else:
        print(f"   Feature importance not available for {model_name}")
        return None

def plot_predictions(y_test, y_test_pred, test_dates, model_name):
    """Plot predicted vs actual values"""
    print(f"\n📈 Creating prediction plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([0, 10], [0, 10], 'r--', lw=2)
    plt.xlabel('Actual Pollen Severity')
    plt.ylabel('Predicted Pollen Severity')
    plt.title(f'{model_name}: Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Time series (last 100 days)
    plt.subplot(2, 2, 2)
    last_100 = slice(-100, None)
    plt.plot(test_dates.iloc[last_100], y_test.iloc[last_100], 'b-', label='Actual', linewidth=2)
    plt.plot(test_dates.iloc[last_100], y_test_pred[last_100], 'r-', label='Predicted', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Pollen Severity')
    plt.title('Time Series: Last 100 Test Days')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    plt.subplot(2, 2, 3)
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of predictions
    plt.subplot(2, 2, 4)
    plt.hist(y_test, bins=11, alpha=0.7, label='Actual', density=True)
    plt.hist(y_test_pred, bins=11, alpha=0.7, label='Predicted', density=True)
    plt.xlabel('Pollen Severity')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pollen_prediction_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   📊 Plots saved to: pollen_prediction_analysis.png")
    plt.show()

def seasonal_analysis(df_clean, y_test, y_test_pred, test_dates):
    """Analyze performance by season"""
    print(f"\n🌱 Seasonal Performance Analysis...")
    
    test_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test,
        'Predicted': y_test_pred
    })
    test_df['Month'] = test_df['Date'].dt.month
    test_df['Season'] = test_df['Month'].map({
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall',
        12: 'Winter', 1: 'Winter', 2: 'Winter'
    })
    
    seasonal_performance = []
    for season in ['Spring', 'Summer', 'Fall', 'Winter']:
        season_data = test_df[test_df['Season'] == season]
        if len(season_data) > 0:
            mae = mean_absolute_error(season_data['Actual'], season_data['Predicted'])
            r2 = r2_score(season_data['Actual'], season_data['Predicted'])
            seasonal_performance.append({
                'Season': season,
                'Samples': len(season_data),
                'MAE': mae,
                'R²': r2,
                'Avg_Actual': season_data['Actual'].mean(),
                'Avg_Predicted': season_data['Predicted'].mean()
            })
    
    seasonal_df = pd.DataFrame(seasonal_performance)
    print(f"   📊 Performance by Season:")
    print(seasonal_df.to_string(index=False, float_format='%.3f'))
    
    return seasonal_df

def save_model_and_results(model, feature_cols, results, model_name):
    """Save model and results for future use"""
    print(f"\n💾 Saving model and results...")
    
    import joblib
    
    # Save model
    model_filename = f'pollen_predictor_{model_name.lower().replace(" ", "_")}.joblib'
    joblib.dump(model, model_filename)
    
    # Save feature columns
    feature_filename = 'model_features.joblib'
    joblib.dump(feature_cols, feature_filename)
    
    # Save results summary
    results_summary = {
        'model_name': model_name,
        'test_mae': results['test_mae'],
        'test_rmse': results['test_rmse'],
        'test_r2': results['test_r2'],
        'feature_count': len(feature_cols),
        'timestamp': datetime.now().isoformat()
    }
    
    results_filename = 'model_results.json'
    import json
    with open(results_filename, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"   ✅ Model saved: {model_filename}")
    print(f"   ✅ Features saved: {feature_filename}")
    print(f"   ✅ Results saved: {results_filename}")

def main():
    """Main execution function"""
    print("🌸 POLLEN INTENSITY PREDICTOR")
    print("=" * 60)
    print("🎯 Goal: Predict daily pollen severity (0-10 scale)")
    print("📊 Model: Random Forest with weather + temporal features")
    print("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        df = load_and_prepare_data()
        
        # Step 2: Create severity scale
        df, thresholds = create_pollen_severity_scale(df)
        
        # Step 3: Engineer features
        df = engineer_features(df)
        
        # Step 4: Select features
        feature_cols = select_features(df)
        
        # Step 5: Prepare train/test split
        X_train, X_test, y_train, y_test, train_dates, test_dates, df_clean = prepare_train_test_split(
            df, feature_cols
        )
        
        # Step 6: Train models
        trained_models, best_model, best_model_name = train_models(X_train, y_train)
        
        # Step 7: Evaluate best model
        results = evaluate_model(best_model, X_train, X_test, y_train, y_test, best_model_name)
        
        # Step 8: Feature importance analysis
        feature_importance = analyze_feature_importance(best_model, feature_cols, best_model_name)
        
        # Step 9: Visualize results
        plot_predictions(y_test, results['test_pred'], test_dates, best_model_name)
        
        # Step 10: Seasonal analysis
        seasonal_performance = seasonal_analysis(df_clean, y_test, results['test_pred'], test_dates)
        
        # Step 11: Save model and results
        save_model_and_results(best_model, feature_cols, results, best_model_name)
        
        print("\n" + "=" * 60)
        print("✅ POLLEN INTENSITY PREDICTOR COMPLETED!")
        print("=" * 60)
        print(f"🏆 Best Model: {best_model_name}")
        print(f"📊 Test MAE: {results['test_mae']:.3f} (0-10 scale)")
        print(f"📈 Test R²: {results['test_r2']:.3f}")
        print(f"🎯 Performance: {'EXCELLENT' if results['test_mae'] <= 1.0 else 'GOOD' if results['test_mae'] <= 2.0 else 'OKAY' if results['test_mae'] <= 3.0 else 'NEEDS IMPROVEMENT'}")
        print("\n🚀 Ready for integration into allergy forecasting system!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())