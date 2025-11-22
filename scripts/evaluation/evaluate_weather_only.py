"""
Compare XGBoost Models: Multitype vs Weather-Only Training
Evaluates performance differences between:
1. Multitype models (trained with all features including pollen history)
2. Weather-only models (trained exclusively with weather features)

This comparison helps determine if training with only weather features
provides any performance benefits for weather-only predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import sys
import os

# Add parent directory to path to import from training scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

from rf import load_and_prepare_data
from xgboost_multitype import (
    create_biological_severity_scales,
    engineer_refined_features,
    select_refined_features,
    comprehensive_metrics
)
from xgboost_weather_only import (
    engineer_features_no_pollen_history,
    select_final_features
)


def evaluate_models_with_weather_features(models_dict, df, feature_cols, model_name):
    """
    Evaluate models using only weather-derivable features
    """
    print(f"\n{'='*70}")
    print(f"🌦️  EVALUATING {model_name.upper()} MODELS WITH WEATHER-ONLY FEATURES")
    print(f"{'='*70}")
    
    # Define weather-derivable features (same logic as in xgboost_multitype.py)
    base_weather_inputs = ['TMAX', 'TMIN', 'AWND', 'PRCP']
    weather_derivable_features = []
    
    for feat in feature_cols:
        # Date-based features
        if any(x in feat for x in ['Month_', 'DOY_', 'Day_of_Week', 'Year_Numeric', 'Month_Numeric']):
            weather_derivable_features.append(feat)
        # Direct weather features
        elif feat in base_weather_inputs:
            weather_derivable_features.append(feat)
        # Weather-derived features
        elif feat in ['TAVG', 'Temp_Range', 'Is_Rainy']:
            weather_derivable_features.append(feat)
        # Weather dynamics
        elif any(x in feat for x in ['Temp_Anomaly', 'TAVG_30day', 'Wind_Percentile', 'VPD', 'Ventilation_Index', 'Shock_Index', 'Days_Since_Rain', 'Rain_Light', 'Rain_Heavy']):
            weather_derivable_features.append(feat)
        # Rainfall/season features
        elif 'Rainfall_Season_Cumsum' in feat or 'Season' in feat:
            weather_derivable_features.append(feat)
        # GDD features
        elif 'GDD' in feat:
            weather_derivable_features.append(feat)
        # Peak proximity
        elif 'Peak_Prox' in feat:
            weather_derivable_features.append(feat)
        # Weather interactions
        elif any(x in feat for x in ['Temp_x_Spring', 'Temp_x_Summer', 'Rain_x_Wind', 'Dry_x_Wind']):
            weather_derivable_features.append(feat)
        # Weather lags/rolling (no pollen needed)
        elif any(x in feat for x in ['TMAX_lag_', 'PRCP_lag_', 'AWND_lag_', 'VPD_lag_']):
            weather_derivable_features.append(feat)
        elif any(x in feat for x in ['Temp_roll_', 'Rain_roll_', 'VPD_roll_']):
            weather_derivable_features.append(feat)
    
    # Remove duplicates and ensure features exist
    weather_derivable_features = [f for f in weather_derivable_features if f in df.columns]
    weather_derivable_features = list(dict.fromkeys(weather_derivable_features))
    
    print(f"📊 Using {len(weather_derivable_features)}/{len(feature_cols)} weather-derivable features")
    
    # Use same 80/20 split as training
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    
    results = {}
    
    for pollen_type, model in models_dict.items():
        severity_col = f'{pollen_type}_Severity'
        
        if severity_col not in test_df.columns:
            continue
        
        print(f"\n🌸 Evaluating {pollen_type}...")
        
        # Prepare test data
        test_clean = test_df.dropna(subset=weather_derivable_features + [severity_col])
        
        # Create feature matrix - fill missing features with 0
        X_test_weather = pd.DataFrame(0, index=test_clean.index, columns=feature_cols)
        
        # Fill in weather-derivable features
        for feat in weather_derivable_features:
            if feat in test_clean.columns:
                X_test_weather[feat] = test_clean[feat]
        
        y_test = test_clean[severity_col]
        
        # Make predictions (convert to numpy to avoid dtype issues)
        y_pred_weather = np.clip(model.predict(X_test_weather.values), 0, 10)
        
        # Calculate metrics
        metrics = comprehensive_metrics(y_test.values, y_pred_weather)
        
        print(f"   MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f} | Acc(±1): {metrics['acc_within_1']:.1%}")
        
        results[pollen_type] = {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'r2': float(metrics['r2']),
            'acc_within_1': float(metrics['acc_within_1']),
            'acc_within_2': float(metrics['acc_within_2']),
            'high_mae': float(metrics['high_mae']) if not np.isnan(metrics['high_mae']) else None,
            'high_count': int(metrics['high_count']),
            'test_samples': len(test_clean),
            'predictions': y_pred_weather.tolist(),
            'actuals': y_test.tolist()
        }
    
    return results


def main():
    print("="*70)
    print("🔬 XGBOOST MODEL COMPARISON: MULTITYPE vs WEATHER-ONLY")
    print("="*70)
    print("Comparing two training approaches:")
    print("  1️⃣  Multitype: Trained with ALL features (including pollen history)")
    print("  2️⃣  Weather-Only: Trained EXCLUSIVELY with weather features")
    print("\nBoth evaluated using ONLY weather-derivable features to simulate")
    print("real-world user input scenario (Date + TMAX + TMIN + AWND + PRCP)")
    print("="*70)
    
    # Load and prepare data
    print("\n[1/5] Loading data...")
    df = load_and_prepare_data()
    print(f"   ✅ Loaded {len(df):,} samples")
    
    # Create severity scales
    print("\n[2/5] Creating severity scales...")
    df = create_biological_severity_scales(df)
    print(f"   ✅ Created severity scales")
    
    # Engineer features for both approaches
    print("\n[3/5] Engineering features...")
    
    # Multitype features (all features)
    df_multitype = engineer_refined_features(df.copy())
    multitype_features = select_refined_features(df_multitype)
    print(f"   ✅ Multitype features: {len(multitype_features)}")
    
    # Weather-only features (no pollen history)
    df_weather_only = engineer_features_no_pollen_history(df.copy())
    weather_only_features = select_final_features(df_weather_only)
    print(f"   ✅ Weather-only features: {len(weather_only_features)}")
    
    # Load trained models
    print("\n[4/5] Loading trained models...")
    pollen_types = ['Tree', 'Grass', 'Weed', 'Ragweed', 'Total_Pollen']
    
    # Load multitype models
    multitype_models = {}
    for pollen_type in pollen_types:
        model_path = f'models/xgboost_{pollen_type.lower()}.joblib'
        if os.path.exists(model_path):
            multitype_models[pollen_type] = joblib.load(model_path)
            print(f"   ✅ Multitype {pollen_type}")
        else:
            print(f"   ⚠️  Multitype {pollen_type} not found")
    
    # Load weather-only models
    weather_only_models = {}
    for pollen_type in pollen_types:
        model_path = f'models/xgboost_{pollen_type.lower()}_bio_v2.joblib'
        if os.path.exists(model_path):
            weather_only_models[pollen_type] = joblib.load(model_path)
            print(f"   ✅ Weather-only {pollen_type}")
        else:
            print(f"   ⚠️  Weather-only {pollen_type} not found")
    
    if not multitype_models and not weather_only_models:
        print("\n❌ No trained models found!")
        print("   Please run training scripts first")
        return 1
    
    print(f"\n   Multitype models: {len(multitype_models)}")
    print(f"   Weather-only models: {len(weather_only_models)}")
    
    # Evaluate both model types
    print("\n[5/5] Evaluating models...")
    
    multitype_results = {}
    weather_only_results = {}
    
    if multitype_models:
        multitype_results = evaluate_models_with_weather_features(
            multitype_models, df_multitype, multitype_features, "Multitype"
        )
    
    if weather_only_models:
        weather_only_results = evaluate_models_with_weather_features(
            weather_only_models, df_weather_only, weather_only_features, "Weather-Only"
        )
    
    # Generate comparison visualization
    print("\n📊 Generating comparison plots...")
    
    if multitype_results and weather_only_results:
        # Get common pollen types
        common_types = [pt for pt in multitype_results.keys() if pt in weather_only_results.keys()]
        
        if common_types:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract metrics for comparison
            multitype_mae = [multitype_results[pt]['mae'] for pt in common_types]
            weather_only_mae = [weather_only_results[pt]['mae'] for pt in common_types]
            multitype_r2 = [multitype_results[pt]['r2'] for pt in common_types]
            weather_only_r2 = [weather_only_results[pt]['r2'] for pt in common_types]
            multitype_acc = [multitype_results[pt]['acc_within_1'] for pt in common_types]
            weather_only_acc = [weather_only_results[pt]['acc_within_1'] for pt in common_types]
            
            x = np.arange(len(common_types))
            width = 0.35
            
            # 1. MAE Comparison
            axes[0, 0].bar(x - width/2, multitype_mae, width, label='Multitype Training', color='#e74c3c', alpha=0.8)
            axes[0, 0].bar(x + width/2, weather_only_mae, width, label='Weather-Only Training', color='#3498db', alpha=0.8)
            axes[0, 0].set_xlabel('Pollen Type')
            axes[0, 0].set_ylabel('MAE')
            axes[0, 0].set_title('Mean Absolute Error Comparison\n(Both evaluated with weather-only features)')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(common_types, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # 2. R² Comparison
            axes[0, 1].bar(x - width/2, multitype_r2, width, label='Multitype Training', color='#e74c3c', alpha=0.8)
            axes[0, 1].bar(x + width/2, weather_only_r2, width, label='Weather-Only Training', color='#3498db', alpha=0.8)
            axes[0, 1].set_xlabel('Pollen Type')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].set_title('R² Score Comparison\n(Both evaluated with weather-only features)')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(common_types, rotation=45, ha='right')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 3. Accuracy Comparison
            axes[1, 0].bar(x - width/2, multitype_acc, width, label='Multitype Training', color='#e74c3c', alpha=0.8)
            axes[1, 0].bar(x + width/2, weather_only_acc, width, label='Weather-Only Training', color='#3498db', alpha=0.8)
            axes[1, 0].set_xlabel('Pollen Type')
            axes[1, 0].set_ylabel('Accuracy (within ±1)')
            axes[1, 0].set_title('Prediction Accuracy Comparison\n(Both evaluated with weather-only features)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(common_types, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].set_ylim(0, 1)
            
            # 4. Performance difference (Weather-Only Training vs Multitype Training)
            mae_improvement = [(m - w) / m * 100 for m, w in zip(multitype_mae, weather_only_mae)]
            
            colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in mae_improvement]
            bars = axes[1, 1].bar(x, mae_improvement, color=colors, alpha=0.7)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1, 1].set_xlabel('Pollen Type')
            axes[1, 1].set_ylabel('MAE Improvement (%)')
            axes[1, 1].set_title('Weather-Only Training Benefit\n(Positive = Weather-Only Training is Better)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(common_types, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, mae_improvement):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1),
                               f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('results/model_training_comparison.png', dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved: results/model_training_comparison.png")
            plt.close()
            
            # Summary statistics
            print(f"\n📊 COMPARISON SUMMARY:")
            print(f"{'='*70}")
            print(f"{'Model Type':<20} {'Avg MAE':<10} {'Avg R²':<10} {'Avg Acc(±1)':<12}")
            print(f"{'-'*70}")
            print(f"{'Multitype Training':<20} {np.mean(multitype_mae):<10.4f} {np.mean(multitype_r2):<10.4f} {np.mean(multitype_acc):<12.1%}")
            print(f"{'Weather-Only Training':<20} {np.mean(weather_only_mae):<10.4f} {np.mean(weather_only_r2):<10.4f} {np.mean(weather_only_acc):<12.1%}")
            print(f"{'-'*70}")
            
            # Calculate overall improvement
            avg_mae_improvement = np.mean(mae_improvement)
            avg_r2_improvement = (np.mean(weather_only_r2) - np.mean(multitype_r2)) / abs(np.mean(multitype_r2)) * 100
            avg_acc_improvement = (np.mean(weather_only_acc) - np.mean(multitype_acc)) / np.mean(multitype_acc) * 100
            
            print(f"\n🎯 WEATHER-ONLY TRAINING BENEFITS:")
            print(f"   MAE Improvement:      {avg_mae_improvement:+.1f}%")
            print(f"   R² Change:            {avg_r2_improvement:+.1f}%")
            print(f"   Accuracy Change:      {avg_acc_improvement:+.1f}%")
            
            # Interpretation
            if avg_mae_improvement > 5:
                print(f"\n✅ SIGNIFICANT BENEFIT: Weather-only training performs notably better!")
                print(f"   Training with only weather features improves weather-only predictions.")
            elif avg_mae_improvement > 0:
                print(f"\n👍 MODEST BENEFIT: Weather-only training shows some improvement.")
                print(f"   There's a small advantage to training with only weather features.")
            elif avg_mae_improvement > -5:
                print(f"\n⚖️  SIMILAR PERFORMANCE: Both training approaches perform comparably.")
                print(f"   No significant difference between training approaches.")
            else:
                print(f"\n⚠️  MULTITYPE ADVANTAGE: Multitype training performs better.")
                print(f"   Training with all features (including pollen history) is superior.")
        
        else:
            print(f"\n⚠️  No common pollen types found between model sets")
    
    elif multitype_results:
        print(f"\n⚠️  Only multitype results available")
    elif weather_only_results:
        print(f"\n⚠️  Only weather-only results available")
    else:
        print(f"\n⚠️  No results to compare")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    comparison_results = {
        'multitype_results': multitype_results,
        'weather_only_results': weather_only_results,
        'comparison_summary': {}
    }
    
    if multitype_results and weather_only_results:
        common_types = [pt for pt in multitype_results.keys() if pt in weather_only_results.keys()]
        if common_types:
            multitype_mae = [multitype_results[pt]['mae'] for pt in common_types]
            weather_only_mae = [weather_only_results[pt]['mae'] for pt in common_types]
            
            comparison_results['comparison_summary'] = {
                'common_pollen_types': common_types,
                'multitype_avg_mae': float(np.mean(multitype_mae)),
                'weather_only_avg_mae': float(np.mean(weather_only_mae)),
                'mae_improvement_percent': float(np.mean([(m - w) / m * 100 for m, w in zip(multitype_mae, weather_only_mae)])),
                'weather_only_training_is_better': bool(np.mean(weather_only_mae) < np.mean(multitype_mae))
            }
    
    with open('results/model_training_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"   ✅ Saved: results/model_training_comparison.json")
    
    print("\n" + "="*70)
    print("✅ MODEL COMPARISON COMPLETE!")
    print("="*70)
    print(f"\n💾 Outputs:")
    print(f"   • results/model_training_comparison.json")
    print(f"   • results/model_training_comparison.png")
    
    print(f"\n🔬 Key Findings:")
    if comparison_results['comparison_summary']:
        summary = comparison_results['comparison_summary']
        if summary['weather_only_training_is_better']:
            print(f"   ✅ Weather-only training is BETTER for weather-only predictions!")
            print(f"      Average MAE improvement: {summary['mae_improvement_percent']:.1f}%")
            print(f"      Multitype: {summary['multitype_avg_mae']:.4f} vs Weather-only: {summary['weather_only_avg_mae']:.4f}")
        else:
            print(f"   ⚖️  Multitype training performs better overall")
            print(f"      Average MAE difference: {summary['mae_improvement_percent']:.1f}%")
            print(f"      Multitype: {summary['multitype_avg_mae']:.4f} vs Weather-only: {summary['weather_only_avg_mae']:.4f}")
        
        print(f"\n💡 Recommendation:")
        if summary['mae_improvement_percent'] > 5:
            print(f"      Use weather-only trained models for weather-only predictions")
        elif summary['mae_improvement_percent'] > -5:
            print(f"      Both approaches work similarly well")
        else:
            print(f"      Multitype training is preferred even for weather-only predictions")
    
    print(f"\n{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user!\n")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        print(f"\n📋 Traceback:")
        traceback.print_exc()
        exit(1)
