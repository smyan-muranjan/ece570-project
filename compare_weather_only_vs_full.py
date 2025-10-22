"""
Weather-Only vs Full-Feature Model Comparison
==============================================

Evaluates model performance in two scenarios:
1. FULL FEATURES: Weather + recent pollen observations (best case)
2. WEATHER ONLY: Weather data only, no pollen history (realistic forecast)

This shows the degradation when pollen monitoring data isn't available.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare test data"""
    print("📂 Loading data...")
    df = pd.read_csv('combined_allergy_weather.csv')
    df['Date_Standard'] = pd.to_datetime(df['Date_Standard'])
    
    # Import feature engineering functions
    print("   Loading feature engineering pipeline...")
    from train_clinical_pollen_models import (
        create_clinical_severity_scale, 
        engineer_features, 
        select_features,
        CLINICAL_THRESHOLDS
    )
    
    # Create clinical severity scales
    df = create_clinical_severity_scale(df)
    
    # Engineer features (same as training)
    df = engineer_features(df)
    features = select_features(df)
    
    # Get test split (same as training - last 20%)
    df_clean = df.dropna(subset=features + ['Tree_Severity', 'Grass_Severity', 'Weed_Severity']).copy()
    split_date = df_clean['Date_Standard'].quantile(0.8)
    test_mask = df_clean['Date_Standard'] > split_date
    
    X_test = df_clean.loc[test_mask, features]
    test_dates = df_clean.loc[test_mask, 'Date_Standard']
    
    y_test = {
        'Tree': df_clean.loc[test_mask, 'Tree_Severity'],
        'Grass': df_clean.loc[test_mask, 'Grass_Severity'],
        'Weed': df_clean.loc[test_mask, 'Weed_Severity'],
        'Tree_count': df_clean.loc[test_mask, 'Tree'],
        'Grass_count': df_clean.loc[test_mask, 'Grass'],
        'Weed_count': df_clean.loc[test_mask, 'Weed']
    }
    
    print(f"   Test set: {len(X_test)} samples")
    print(f"   Date range: {test_dates.min().date()} to {test_dates.max().date()}")
    
    return X_test, y_test, features, test_dates

def identify_pollen_features(features):
    """Identify which features depend on pollen observations"""
    pollen_features = []
    weather_features = []
    
    pollen_keywords = ['Pollen', 'pollen', 'Tree_lag', 'Grass_lag', 'Weed_lag', 
                       'Tree_roll', 'Grass_roll', 'Weed_roll',
                       'Tree_Severity_lag', 'Grass_Severity_lag', 'Weed_Severity_lag',
                       'Tree_acceleration', 'Grass_acceleration', 'Weed_acceleration',
                       'Tree_trend', 'Grass_trend', 'Weed_trend',
                       'Tree_volatility', 'Grass_volatility', 'Weed_volatility',
                       'Tree_spike', 'Grass_spike', 'Weed_spike',
                       'Tree_x_', 'Grass_x_', 'Weed_x_',
                       'increasing', 'decreasing']
    
    for feat in features:
        if any(keyword in feat for keyword in pollen_keywords):
            pollen_features.append(feat)
        else:
            weather_features.append(feat)
    
    return pollen_features, weather_features

def create_weather_only_features(X_test, pollen_features, y_test, test_dates):
    """
    Create weather-only feature set by replacing pollen features with fallbacks
    
    Fallback strategies:
    1. Lags: Use seasonal median for that day-of-year
    2. Rolling averages: Use seasonal median
    3. Interactions: Set to 0 (no pollen to interact with weather)
    4. Momentum: Set to 0 (no trend without history)
    """
    print("\n🌤️  Creating weather-only feature set...")
    
    X_weather_only = X_test.copy()
    
    # Calculate seasonal medians from historical data (training period)
    split_date = test_dates.min()
    df_full = pd.read_csv('combined_allergy_weather.csv')
    df_full['Date_Standard'] = pd.to_datetime(df_full['Date_Standard'])
    df_full['DOY'] = df_full['Date_Standard'].dt.dayofyear
    
    # Get training data for seasonal estimates
    train_data = df_full[df_full['Date_Standard'] < split_date].copy()
    
    # Calculate seasonal medians (30-day window around each DOY)
    seasonal_medians = {}
    for doy in range(1, 366):
        window_mask = (train_data['DOY'] >= doy - 15) & (train_data['DOY'] <= doy + 15)
        seasonal_medians[doy] = {
            'Tree': train_data.loc[window_mask, 'Tree'].median(),
            'Grass': train_data.loc[window_mask, 'Grass'].median(),
            'Weed': train_data.loc[window_mask, 'Weed'].median()
        }
    
    # Map test dates to DOY
    test_doy = test_dates.dt.dayofyear.values
    
    print(f"   Replacing {len(pollen_features)} pollen-dependent features...")
    
    replacements = {
        'lag': 0,
        'roll': 0,
        'acceleration': 0,
        'trend': 0,
        'volatility': 0,
        'spike': 0,
        'x_': 0,  # Interactions
        'increasing': 0,
        'decreasing': 0,
        'Severity': 1  # Default to "Low" severity (level 1)
    }
    
    for feat in pollen_features:
        # Determine which pollen type
        pollen_type = None
        if 'Tree' in feat:
            pollen_type = 'Tree'
        elif 'Grass' in feat:
            pollen_type = 'Grass'
        elif 'Weed' in feat:
            pollen_type = 'Weed'
        
        # Use seasonal median for lags and rolling averages
        if 'lag_' in feat or 'roll_' in feat:
            if pollen_type:
                # Use seasonal median for each test sample
                fallback_values = np.array([seasonal_medians[doy][pollen_type] for doy in test_doy])
                X_weather_only[feat] = fallback_values
            else:
                X_weather_only[feat] = 0
        
        # Zero out momentum and interaction features
        else:
            for keyword, value in replacements.items():
                if keyword in feat:
                    X_weather_only[feat] = value
                    break
    
    print(f"   ✅ Weather-only features ready")
    print(f"   • Weather features: {len(X_test.columns) - len(pollen_features)}")
    print(f"   • Pollen features replaced: {len(pollen_features)}")
    
    return X_weather_only

def evaluate_model(model_path, X_full, X_weather, y_true, pollen_type):
    """Evaluate model with full features vs weather-only"""
    print(f"\n📊 Evaluating {pollen_type} model...")
    
    # Load model
    try:
        model = joblib.load(model_path)
    except:
        print(f"   ⚠️  Model not found: {model_path}")
        return None
    
    # Predictions with full features
    pred_full = model.predict(X_full)
    mae_full = mean_absolute_error(y_true, pred_full)
    r2_full = r2_score(y_true, pred_full)
    within_1_full = np.mean(np.abs(y_true - pred_full) <= 1)
    
    # Predictions with weather-only
    pred_weather = model.predict(X_weather)
    mae_weather = mean_absolute_error(y_true, pred_weather)
    r2_weather = r2_score(y_true, pred_weather)
    within_1_weather = np.mean(np.abs(y_true - pred_weather) <= 1)
    
    # Calculate degradation
    mae_degradation = ((mae_weather - mae_full) / mae_full) * 100
    r2_degradation = ((r2_full - r2_weather) / r2_full) * 100
    acc_degradation = ((within_1_full - within_1_weather) / within_1_full) * 100
    
    results = {
        'full_features': {
            'mae': float(mae_full),
            'r2': float(r2_full),
            'accuracy_within_1': float(within_1_full),
            'predictions': pred_full
        },
        'weather_only': {
            'mae': float(mae_weather),
            'r2': float(r2_weather),
            'accuracy_within_1': float(within_1_weather),
            'predictions': pred_weather
        },
        'degradation': {
            'mae_pct': float(mae_degradation),
            'r2_pct': float(r2_degradation),
            'accuracy_pct': float(acc_degradation)
        }
    }
    
    print(f"   Full Features:    MAE={mae_full:.4f}, R²={r2_full:.4f}, Acc={within_1_full:.1%}")
    print(f"   Weather Only:     MAE={mae_weather:.4f}, R²={r2_weather:.4f}, Acc={within_1_weather:.1%}")
    print(f"   Degradation:      MAE +{mae_degradation:.1f}%, R² -{r2_degradation:.1f}%, Acc -{acc_degradation:.1f}%")
    
    return results

def create_comparison_visualization(all_results, y_test, test_dates):
    """Create visualization comparing full vs weather-only predictions"""
    print("\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance: Full Features vs Weather-Only', fontsize=16, fontweight='bold')
    
    pollen_types = ['Tree', 'Grass', 'Weed']
    
    for idx, pollen_type in enumerate(pollen_types):
        ax = axes[idx // 2, idx % 2]
        
        results = all_results[pollen_type]
        if results is None:
            continue
        
        y_true = y_test[pollen_type].values
        pred_full = results['full_features']['predictions']
        pred_weather = results['weather_only']['predictions']
        
        # Time series plot
        dates = test_dates.values
        
        ax.plot(dates, y_true, 'ko-', label='Actual', alpha=0.6, linewidth=2, markersize=4)
        ax.plot(dates, pred_full, 'b.-', label='Full Features', alpha=0.7, linewidth=1.5, markersize=3)
        ax.plot(dates, pred_weather, 'r.--', label='Weather Only', alpha=0.7, linewidth=1.5, markersize=3)
        
        ax.set_title(f'{pollen_type} Pollen Severity', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Clinical Severity (0-4)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 4.5)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add performance metrics as text
        mae_full = results['full_features']['mae']
        mae_weather = results['weather_only']['mae']
        r2_full = results['full_features']['r2']
        r2_weather = results['weather_only']['r2']
        
        textstr = f'Full: MAE={mae_full:.3f}, R²={r2_full:.3f}\nWeather: MAE={mae_weather:.3f}, R²={r2_weather:.3f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Summary comparison in 4th subplot
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "📊 PERFORMANCE SUMMARY\n\n"
    
    for pollen_type in pollen_types:
        results = all_results[pollen_type]
        if results is None:
            continue
        
        summary_text += f"{pollen_type} Pollen:\n"
        summary_text += f"  Full Features:  MAE {results['full_features']['mae']:.3f}, R² {results['full_features']['r2']:.3f}\n"
        summary_text += f"  Weather Only:   MAE {results['weather_only']['mae']:.3f}, R² {results['weather_only']['r2']:.3f}\n"
        summary_text += f"  Degradation:    +{results['degradation']['mae_pct']:.1f}% MAE\n\n"
    
    summary_text += "\n💡 KEY INSIGHT:\n"
    summary_text += "Weather-only forecasting shows\n"
    summary_text += "moderate degradation but remains\n"
    summary_text += "usable for first-time users or\n"
    summary_text += "areas without pollen monitoring."
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('weather_vs_full_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: weather_vs_full_comparison.png")
    
    plt.close()

def main():
    print("="*80)
    print("🌤️  WEATHER-ONLY vs FULL-FEATURE MODEL COMPARISON")
    print("="*80)
    print("Evaluating model performance degradation without pollen observations")
    print("="*80)
    
    # Load test data
    X_test, y_test, features, test_dates = load_data()
    
    # Identify pollen vs weather features
    pollen_features, weather_features = identify_pollen_features(features)
    
    print(f"\n📋 Feature Breakdown:")
    print(f"   Total features: {len(features)}")
    print(f"   Weather-only features: {len(weather_features)}")
    print(f"   Pollen-dependent features: {len(pollen_features)}")
    print(f"   Pollen dependency: {len(pollen_features)/len(features)*100:.1f}%")
    
    # Create weather-only feature set
    X_weather_only = create_weather_only_features(X_test, pollen_features, y_test, test_dates)
    
    # Evaluate each model
    print("\n" + "="*80)
    print("🔬 MODEL EVALUATION")
    print("="*80)
    
    models = {
        'Tree': 'pollen_predictor_tree_severity.joblib',
        'Grass': 'pollen_predictor_grass_severity.joblib',
        'Weed': 'pollen_predictor_weed_severity.joblib'
    }
    
    all_results = {}
    
    for pollen_type, model_path in models.items():
        results = evaluate_model(model_path, X_test, X_weather_only, y_test[pollen_type], pollen_type)
        all_results[pollen_type] = results
    
    # Summary table
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON TABLE")
    print("="*80)
    
    print(f"\n{'Pollen Type':<12} {'Scenario':<15} {'MAE':<10} {'R²':<10} {'Acc(±1)':<10}")
    print("-" * 80)
    
    for pollen_type in ['Tree', 'Grass', 'Weed']:
        results = all_results[pollen_type]
        if results is None:
            continue
        
        # Full features
        print(f"{pollen_type:<12} {'Full Features':<15} {results['full_features']['mae']:<10.4f} "
              f"{results['full_features']['r2']:<10.4f} {results['full_features']['accuracy_within_1']:<10.1%}")
        
        # Weather only
        print(f"{'':<12} {'Weather Only':<15} {results['weather_only']['mae']:<10.4f} "
              f"{results['weather_only']['r2']:<10.4f} {results['weather_only']['accuracy_within_1']:<10.1%}")
        
        # Degradation
        print(f"{'':<12} {'Degradation':<15} {results['degradation']['mae_pct']:>+9.1f}% "
              f"{results['degradation']['r2_pct']:>+9.1f}% {results['degradation']['accuracy_pct']:>+9.1f}%")
        print()
    
    # Key insights
    print("="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)
    
    avg_mae_degradation = np.mean([r['degradation']['mae_pct'] for r in all_results.values() if r])
    avg_r2_degradation = np.mean([r['degradation']['r2_pct'] for r in all_results.values() if r])
    
    print(f"""
Average Performance Degradation (Weather-Only):
• MAE increases by {avg_mae_degradation:.1f}%
• R² decreases by {avg_r2_degradation:.1f}%

Interpretation:
• Models remain {100-avg_mae_degradation:.0f}% as accurate without pollen data
• Weather-only forecasting is viable for:
  - First-time app users (no history)
  - Areas without pollen monitoring stations
  - Multi-day ahead forecasts

Recommendation:
• Use FULL FEATURES when available (best accuracy)
• Gracefully degrade to WEATHER-ONLY when needed
• Collect pollen observations to improve future predictions
    """)
    
    # Create visualization
    create_comparison_visualization(all_results, y_test, test_dates)
    
    # Save results
    print("\n💾 Saving results...")
    
    save_results = {
        'feature_breakdown': {
            'total_features': len(features),
            'weather_features': len(weather_features),
            'pollen_features': len(pollen_features)
        },
        'model_performance': {k: {
            'full_features': v['full_features'],
            'weather_only': v['weather_only'],
            'degradation': v['degradation']
        } for k, v in all_results.items() if v},
        'summary': {
            'avg_mae_degradation_pct': float(avg_mae_degradation),
            'avg_r2_degradation_pct': float(avg_r2_degradation)
        },
        'evaluated_at': str(datetime.now())
    }
    
    with open('weather_only_comparison.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print("   ✅ weather_only_comparison.json")
    
    print("\n✅ COMPARISON COMPLETE!")

if __name__ == "__main__":
    main()
