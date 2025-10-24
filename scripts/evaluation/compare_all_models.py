"""
Comprehensive Model Comparison Script
Compares all trained models using their saved results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# LOAD MODEL RESULTS
# ============================================================================

def load_model_results(results_dir='results'):
    """Load all model results from JSON files"""
    print("="*80)
    print("📊 LOADING MODEL RESULTS")
    print("="*80)
    
    results_path = Path(results_dir)
    model_results = {}
    
    # Define model mappings (filename -> display name)
    model_files = {
        'rf.json': 'Random Forest (Baseline)',
        'rf_enhanced_features.json': 'Random Forest (Enhanced)',
        'rf_weather_only.json': 'Random Forest (Weather Only)',
        'xgboost.json': 'XGBoost (Baseline)',
        'xgboost_enhanced_features.json': 'XGBoost (Enhanced)',
        'weather_only.json': 'Weather Only Model',
        'model_results.json': 'Legacy Model',
        'model_results_xgboost.json': 'XGBoost Advanced',
    }
    
    for filename, display_name in model_files.items():
        filepath = results_path / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    model_results[display_name] = data
                    print(f"   ✅ Loaded: {display_name}")
            except Exception as e:
                print(f"   ⚠️  Error loading {filename}: {e}")
        else:
            print(f"   ⏭️  Skipped: {display_name} (file not found)")
    
    # Special handling for xgboost_multitype.json (extract Total_Pollen only)
    multitype_path = results_path / 'xgboost_multitype.json'
    if multitype_path.exists():
        try:
            with open(multitype_path, 'r') as f:
                multitype_data = json.load(f)
                # Extract only Total_Pollen as it's comparable to other models
                if 'Total_Pollen' in multitype_data:
                    model_results['XGBoost Multi-Type (Total Pollen)'] = multitype_data['Total_Pollen']
                    print(f"   ✅ Loaded: XGBoost Multi-Type (Total Pollen)")
        except Exception as e:
            print(f"   ⚠️  Error loading xgboost_multitype.json: {e}")
    
    print(f"\n   Total models loaded: {len(model_results)}")
    return model_results


# ============================================================================
# NORMALIZE METRICS
# ============================================================================

def normalize_metrics(model_results):
    """
    Normalize different JSON structures to common format
    Some models use 'mae', others use 'test_mae', etc.
    """
    print("\n" + "="*80)
    print("🔄 NORMALIZING METRICS")
    print("="*80)
    
    normalized = {}
    
    for model_name, data in model_results.items():
        normalized_data = {}
        
        # MAE
        normalized_data['mae'] = (
            data.get('mae') or 
            data.get('test_mae') or 
            data.get('MAE') or 
            None
        )
        
        # RMSE
        normalized_data['rmse'] = (
            data.get('rmse') or 
            data.get('test_rmse') or 
            data.get('RMSE') or 
            None
        )
        
        # R²
        normalized_data['r2'] = (
            data.get('r2') or 
            data.get('test_r2') or 
            data.get('R2') or 
            None
        )
        
        # Additional metrics
        normalized_data['acc_within_1'] = (
            data.get('acc_within_1') or 
            data.get('accuracy_within_1') or 
            None
        )
        
        normalized_data['acc_within_2'] = (
            data.get('acc_within_2') or 
            data.get('accuracy_within_2') or 
            None
        )
        
        # Training time
        normalized_data['train_time'] = (
            data.get('train_time') or 
            data.get('training_time') or 
            None
        )
        
        # Feature count
        normalized_data['feature_count'] = (
            data.get('feature_count') or 
            data.get('n_features') or 
            None
        )
        
        # High severity MAE
        normalized_data['high_mae'] = (
            data.get('high_mae') or 
            data.get('high_severity_mae') or 
            None
        )
        
        normalized[model_name] = normalized_data
    
    # Remove models with no valid metrics
    normalized = {k: v for k, v in normalized.items() 
                  if v['mae'] is not None and v['r2'] is not None}
    
    print(f"   ✅ Normalized {len(normalized)} models with valid metrics")
    return normalized


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def create_comparison_table(normalized_results):
    """Create a comprehensive comparison table"""
    print("\n" + "="*80)
    print("📋 MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Convert to DataFrame for easy sorting and display
    df_data = []
    for model_name, metrics in normalized_results.items():
        row = {
            'Model': model_name,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R²': metrics['r2'],
            'Acc (±1)': metrics.get('acc_within_1'),
            'Acc (±2)': metrics.get('acc_within_2'),
            'High MAE': metrics.get('high_mae'),
            'Features': metrics.get('feature_count'),
            'Train Time': metrics.get('train_time')
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Sort by MAE (lower is better)
    df = df.sort_values('MAE')
    
    # Display main metrics
    print("\n🎯 Main Performance Metrics (sorted by MAE):")
    print("="*80)
    
    # Format and display
    display_df = df[['Model', 'MAE', 'RMSE', 'R²']].copy()
    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    print(display_df.to_string(index=False))
    
    # Additional metrics
    print("\n\n📊 Additional Metrics:")
    print("="*80)
    
    add_cols = ['Model', 'Acc (±1)', 'Acc (±2)', 'High MAE']
    display_add = df[add_cols].copy()
    display_add['Acc (±1)'] = display_add['Acc (±1)'].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
    )
    display_add['Acc (±2)'] = display_add['Acc (±2)'].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
    )
    display_add['High MAE'] = display_add['High MAE'].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )
    
    print(display_add.to_string(index=False))
    
    # Model complexity and efficiency
    print("\n\n⚙️ Model Complexity & Efficiency:")
    print("="*80)
    
    eff_cols = ['Model', 'Features', 'Train Time']
    display_eff = df[eff_cols].copy()
    display_eff['Features'] = display_eff['Features'].apply(
        lambda x: f"{int(x)}" if pd.notna(x) else "N/A"
    )
    display_eff['Train Time'] = display_eff['Train Time'].apply(
        lambda x: f"{x:.2f}s" if pd.notna(x) else "N/A"
    )
    
    print(display_eff.to_string(index=False))
    
    return df


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def statistical_analysis(df):
    """Perform statistical analysis on model performance"""
    print("\n" + "="*80)
    print("📈 STATISTICAL ANALYSIS")
    print("="*80)
    
    # Best model
    best_mae_model = df.loc[df['MAE'].idxmin(), 'Model']
    best_mae = df['MAE'].min()
    
    best_r2_model = df.loc[df['R²'].idxmax(), 'Model']
    best_r2 = df['R²'].max()
    
    print(f"\n🏆 Best Models:")
    print(f"   Best MAE:  {best_mae_model} ({best_mae:.4f})")
    print(f"   Best R²:   {best_r2_model} ({best_r2:.4f})")
    
    # Performance spread
    mae_range = df['MAE'].max() - df['MAE'].min()
    mae_std = df['MAE'].std()
    
    print(f"\n📊 Performance Distribution:")
    print(f"   MAE Range:     {mae_range:.4f}")
    print(f"   MAE Std Dev:   {mae_std:.4f}")
    print(f"   MAE Mean:      {df['MAE'].mean():.4f}")
    print(f"   MAE Median:    {df['MAE'].median():.4f}")
    
    # Improvement analysis
    if len(df) >= 2:
        baseline_mae = df['MAE'].max()
        best_mae = df['MAE'].min()
        improvement = ((baseline_mae - best_mae) / baseline_mae) * 100
        
        print(f"\n🎯 Improvement Analysis:")
        print(f"   Baseline (worst) MAE:  {baseline_mae:.4f}")
        print(f"   Best MAE:              {best_mae:.4f}")
        print(f"   Improvement:           {improvement:.1f}%")
    
    # Model categories
    rf_models = df[df['Model'].str.contains('Random Forest', na=False)]
    xgb_models = df[df['Model'].str.contains('XGBoost', na=False)]
    
    if len(rf_models) > 0:
        print(f"\n🌲 Random Forest Models:")
        print(f"   Count:     {len(rf_models)}")
        print(f"   Best MAE:  {rf_models['MAE'].min():.4f}")
        print(f"   Avg MAE:   {rf_models['MAE'].mean():.4f}")
    
    if len(xgb_models) > 0:
        print(f"\n🚀 XGBoost Models:")
        print(f"   Count:     {len(xgb_models)}")
        print(f"   Best MAE:  {xgb_models['MAE'].min():.4f}")
        print(f"   Avg MAE:   {xgb_models['MAE'].mean():.4f}")
    
    if len(rf_models) > 0 and len(xgb_models) > 0:
        rf_best = rf_models['MAE'].min()
        xgb_best = xgb_models['MAE'].min()
        
        print(f"\n⚔️  Algorithm Comparison:")
        if xgb_best < rf_best:
            improvement = ((rf_best - xgb_best) / rf_best) * 100
            print(f"   XGBoost is {improvement:.1f}% better than Random Forest")
        else:
            improvement = ((xgb_best - rf_best) / xgb_best) * 100
            print(f"   Random Forest is {improvement:.1f}% better than XGBoost")


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_comparison_plots(df, output_path='results/model_comparison.png'):
    """Create comprehensive comparison visualizations"""
    print("\n" + "="*80)
    print("📊 CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. MAE Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, :2])
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(df)))
    bars = ax1.barh(df['Model'], df['MAE'], color=colors)
    ax1.set_xlabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison - MAE', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['MAE'])):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # 2. R² Comparison (Bar Chart)
    ax2 = fig.add_subplot(gs[0, 2])
    colors_r2 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    bars = ax2.barh(df['Model'], df['R²'], color=colors_r2)
    ax2.set_xlabel('R² Score', fontsize=11, fontweight='bold')
    ax2.set_title('R² Comparison', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(df['R²'].min() - 0.05, df['R²'].max() + 0.05)
    
    # 3. MAE vs R² Scatter
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(df['MAE'], df['R²'], s=200, c=range(len(df)), 
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    ax3.set_xlabel('MAE (lower is better)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('R² (higher is better)', fontsize=11, fontweight='bold')
    ax3.set_title('MAE vs R² Trade-off', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add annotations
    for idx, row in df.iterrows():
        ax3.annotate(row['Model'].split('(')[0].strip(), 
                    (row['MAE'], row['R²']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # 4. RMSE Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(range(len(df)), df['RMSE'], color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([m.split('(')[0].strip() for m in df['Model']], 
                        rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax4.set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Accuracy Metrics (if available)
    if df['Acc (±1)'].notna().any():
        ax5 = fig.add_subplot(gs[1, 2])
        acc_data = df[['Model', 'Acc (±1)', 'Acc (±2)']].dropna()
        if len(acc_data) > 0:
            x = np.arange(len(acc_data))
            width = 0.35
            ax5.bar(x - width/2, acc_data['Acc (±1)'] * 100, width, 
                   label='±1 level', color='skyblue', alpha=0.8, edgecolor='black')
            ax5.bar(x + width/2, acc_data['Acc (±2)'] * 100, width, 
                   label='±2 levels', color='lightcoral', alpha=0.8, edgecolor='black')
            ax5.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
            ax5.set_title('Classification Accuracy', fontsize=12, fontweight='bold')
            ax5.set_xticks(x)
            ax5.set_xticklabels([m.split('(')[0].strip() for m in acc_data['Model']], 
                               rotation=45, ha='right', fontsize=9)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Feature Count vs Performance
    if df['Features'].notna().any():
        ax6 = fig.add_subplot(gs[2, 0])
        feature_df = df[['Model', 'Features', 'MAE', 'R²']].dropna(subset=['Features'])
        if len(feature_df) > 0:
            scatter = ax6.scatter(feature_df['Features'], feature_df['MAE'], 
                                s=200, c=feature_df['R²'], cmap='RdYlGn',
                                alpha=0.7, edgecolors='black', linewidth=2)
            ax6.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
            ax6.set_ylabel('MAE', fontsize=11, fontweight='bold')
            ax6.set_title('Model Complexity vs Performance', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax6)
            cbar.set_label('R²', fontsize=10)
            
            # Add annotations
            for idx, row in feature_df.iterrows():
                ax6.annotate(row['Model'].split('(')[0].strip(), 
                            (row['Features'], row['MAE']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7)
    
    # 7. Training Time Comparison (if available)
    if df['Train Time'].notna().any():
        ax7 = fig.add_subplot(gs[2, 1])
        time_df = df[['Model', 'Train Time']].dropna(subset=['Train Time'])
        if len(time_df) > 0:
            bars = ax7.barh(time_df['Model'], time_df['Train Time'], 
                           color='coral', alpha=0.7, edgecolor='black')
            ax7.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
            ax7.set_title('Training Efficiency', fontsize=12, fontweight='bold')
            ax7.invert_yaxis()
            ax7.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, val in zip(bars, time_df['Train Time']):
                ax7.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}s', 
                        va='center', ha='left', fontsize=9)
    
    # 8. Performance Summary Table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = f"""
    📊 PERFORMANCE SUMMARY
    
    Best MAE:  {df['MAE'].min():.4f}
    Worst MAE: {df['MAE'].max():.4f}
    Avg MAE:   {df['MAE'].mean():.4f}
    
    Best R²:   {df['R²'].max():.4f}
    Worst R²:  {df['R²'].min():.4f}
    Avg R²:    {df['R²'].mean():.4f}
    
    Models Compared: {len(df)}
    
    🏆 Winner:
    {df.loc[df['MAE'].idxmin(), 'Model']}
    """
    
    ax8.text(0.1, 0.5, summary_text, fontsize=11, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    fig.suptitle('Comprehensive Model Comparison - Pollen Prediction', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved comparison plot: {output_path}")
    
    plt.close()


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_comparison_results(df, output_path='results/model_comparison.csv'):
    """Save comparison results to CSV"""
    print("\n" + "="*80)
    print("💾 SAVING COMPARISON RESULTS")
    print("="*80)
    
    df.to_csv(output_path, index=False)
    print(f"   ✅ Saved CSV: {output_path}")
    
    # Save summary JSON
    summary = {
        'total_models': len(df),
        'best_model': df.loc[df['MAE'].idxmin(), 'Model'],
        'best_mae': float(df['MAE'].min()),
        'best_r2': float(df['R²'].max()),
        'mae_stats': {
            'mean': float(df['MAE'].mean()),
            'median': float(df['MAE'].median()),
            'std': float(df['MAE'].std()),
            'min': float(df['MAE'].min()),
            'max': float(df['MAE'].max())
        },
        'r2_stats': {
            'mean': float(df['R²'].mean()),
            'median': float(df['R²'].median()),
            'std': float(df['R²'].std()),
            'min': float(df['R²'].min()),
            'max': float(df['R²'].max())
        }
    }
    
    summary_path = 'results/comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Saved summary: {summary_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("🌸 POLLEN PREDICTION MODEL COMPARISON")
    print("="*80)
    print("Comparing all trained models across key performance metrics")
    print("="*80)
    
    # Load results
    model_results = load_model_results()
    
    if len(model_results) == 0:
        print("\n❌ No model results found!")
        print("   Make sure you have trained models and their results are in the 'results/' directory")
        return 1
    
    # Normalize metrics
    normalized_results = normalize_metrics(model_results)
    
    if len(normalized_results) == 0:
        print("\n❌ No valid metrics found in model results!")
        return 1
    
    # Create comparison table
    df = create_comparison_table(normalized_results)
    
    # Statistical analysis
    statistical_analysis(df)
    
    # Create visualizations
    create_comparison_plots(df)
    
    # Save results
    save_comparison_results(df)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)
    
    best_model = df.loc[df['MAE'].idxmin(), 'Model']
    best_mae = df['MAE'].min()
    best_r2 = df.loc[df['MAE'].idxmin(), 'R²']
    
    print(f"\n🏆 WINNER: {best_model}")
    print(f"   MAE: {best_mae:.4f}")
    print(f"   R²:  {best_r2:.4f}")
    
    print("\n📁 Output files:")
    print("   • results/model_comparison.png  (visualizations)")
    print("   • results/model_comparison.csv  (detailed table)")
    print("   • results/comparison_summary.json  (summary stats)")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Comparison interrupted!")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
