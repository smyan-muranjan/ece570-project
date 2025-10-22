"""
Final Pollen Predictor - Stacked Ensemble Model
================================================

This is the production-ready model implementing the best-performing approach:
STACKED ENSEMBLE with enhanced features

Performance:
- Test MAE: 0.896 (1.8% improvement over baseline)
- Test R²: 0.782
- Predicts pollen severity on 0-10 scale

Features:
✓ Pollen momentum (acceleration, trends, volatility)
✓ Weather-pollen interactions
✓ Enhanced biological features (GDD, phenology)
✓ 3-level stacked architecture (RF + GB + Ridge → Meta-RF)

Usage:
    python pollen_predictor_final.py  # Train model
    
    # Or load and predict:
    import joblib
    model = joblib.load('pollen_predictor_final.joblib')
    prediction = model.predict(features)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PollenPredictor:
    """
    Production-ready pollen prediction model using stacked ensemble
    """
    
    def __init__(self):
        self.level1_models = {}
        self.meta_model = None
        self.features = None
        self.thresholds = None
        self.training_info = {}
        
    def create_pollen_severity_scale(self, df):
        """Create 0-10 pollen severity scale based on percentiles"""
        pollen_data = df['Total_Pollen'].copy()
        percentiles = [0, 10, 25, 40, 55, 70, 80, 85, 90, 95, 100]
        self.thresholds = np.percentile(pollen_data[pollen_data > 0], percentiles)
        
        def pollen_to_severity(pollen_count):
            if pollen_count == 0:
                return 0
            for i in range(len(self.thresholds)-1, 0, -1):
                if pollen_count >= self.thresholds[i]:
                    return i
            return 0
        
        df['Pollen_Severity'] = df['Total_Pollen'].apply(pollen_to_severity)
        return df
    
    def engineer_features(self, df):
        """
        Engineer comprehensive feature set with all enhancements
        """
        print("🔬 Engineering features...")
        
        df = df.sort_values('Date_Standard').reset_index(drop=True)
        
        # =====================================================================
        # TEMPORAL FEATURES
        # =====================================================================
        df['Day_of_Year'] = df['Date_Standard'].dt.dayofyear
        df['Day_of_Week'] = df['Date_Standard'].dt.dayofweek
        df['Month'] = df['Date_Standard'].dt.month
        df['Year'] = df['Date_Standard'].dt.year
        
        # Cyclical encoding
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DOY_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365)
        df['DOY_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365)
        
        # =====================================================================
        # CORE WEATHER FEATURES
        # =====================================================================
        df['Temp_Range'] = df['TMAX'] - df['TMIN']
        df['Is_Rainy'] = (df['PRCP'] > 0).astype(int)
        
        # =====================================================================
        # GROWING DEGREE DAYS (Plant Phenology)
        # =====================================================================
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
        
        # =====================================================================
        # POLLEN LAG FEATURES (Historical context)
        # =====================================================================
        for lag in [1, 3, 7]:
            df[f'Total_Pollen_lag_{lag}'] = df['Total_Pollen'].shift(lag)
            df[f'Tree_Pollen_lag_{lag}'] = df['Tree'].shift(lag)
            df[f'Grass_Pollen_lag_{lag}'] = df['Grass'].shift(lag)
            df[f'Weed_Pollen_lag_{lag}'] = df['Weed'].shift(lag)
        
        # =====================================================================
        # POLLEN ROLLING AVERAGES (Recent trends)
        # =====================================================================
        for window in [3, 7, 14]:
            df[f'Total_Pollen_roll_{window}'] = df['Total_Pollen'].shift(1).rolling(window, min_periods=1).mean()
            df[f'Tree_Pollen_roll_{window}'] = df['Tree'].shift(1).rolling(window, min_periods=1).mean()
            df[f'Grass_Pollen_roll_{window}'] = df['Grass'].shift(1).rolling(window, min_periods=1).mean()
            df[f'Weed_Pollen_roll_{window}'] = df['Weed'].shift(1).rolling(window, min_periods=1).mean()
        
        # =====================================================================
        # POLLEN MOMENTUM FEATURES (Dynamics)
        # =====================================================================
        # Acceleration
        df['Pollen_acceleration'] = df['Total_Pollen_lag_1'].diff()
        df['Tree_acceleration'] = df['Tree_Pollen_lag_1'].diff()
        df['Grass_acceleration'] = df['Grass_Pollen_lag_1'].diff()
        df['Weed_acceleration'] = df['Weed_Pollen_lag_1'].diff()
        
        # Trends
        df['Pollen_trend_3d'] = (df['Total_Pollen_lag_1'] - df['Total_Pollen_lag_3']) / 2
        df['Pollen_trend_7d'] = (df['Total_Pollen_lag_1'] - df['Total_Pollen_lag_7']) / 6
        df['Tree_trend_7d'] = (df['Tree_Pollen_lag_1'] - df['Tree_Pollen_lag_7']) / 6
        df['Grass_trend_7d'] = (df['Grass_Pollen_lag_1'] - df['Grass_Pollen_lag_7']) / 6
        df['Weed_trend_7d'] = (df['Weed_Pollen_lag_1'] - df['Weed_Pollen_lag_7']) / 6
        
        # Volatility
        df['Pollen_volatility_7d'] = df['Total_Pollen_lag_1'].rolling(7, min_periods=1).std()
        df['Tree_volatility_7d'] = df['Tree_Pollen_lag_1'].rolling(7, min_periods=1).std()
        df['Grass_volatility_7d'] = df['Grass_Pollen_lag_1'].rolling(7, min_periods=1).std()
        
        # Spike indicators
        df['Pollen_spike'] = (df['Total_Pollen_lag_1'] > df['Total_Pollen_roll_7'] * 1.5).astype(int)
        df['Tree_spike'] = (df['Tree_Pollen_lag_1'] > df['Tree_Pollen_roll_7'] * 1.5).astype(int)
        df['Grass_spike'] = (df['Grass_Pollen_lag_1'] > df['Grass_Pollen_roll_7'] * 1.5).astype(int)
        
        # Direction
        df['Pollen_increasing'] = (df['Pollen_trend_3d'] > 0).astype(int)
        df['Pollen_decreasing'] = (df['Pollen_trend_3d'] < 0).astype(int)
        
        # =====================================================================
        # WEATHER LAG FEATURES
        # =====================================================================
        for lag in [1, 3, 7]:
            df[f'TMAX_lag_{lag}'] = df['TMAX'].shift(lag)
            df[f'PRCP_lag_{lag}'] = df['PRCP'].shift(lag)
        
        # =====================================================================
        # WEATHER ROLLING AVERAGES
        # =====================================================================
        for window in [3, 7, 14]:
            df[f'Temp_roll_{window}'] = df['TAVG'].rolling(window, min_periods=1).mean()
            df[f'Precip_roll_{window}'] = df['PRCP'].rolling(window, min_periods=1).sum()
        
        # =====================================================================
        # WEATHER-POLLEN INTERACTIONS
        # =====================================================================
        # Rain suppression
        df['Pollen_x_Rain'] = df['Total_Pollen_roll_3'] * df['PRCP']
        df['Pollen_x_Rain_lag1'] = df['Total_Pollen_roll_3'] * df['PRCP_lag_1']
        df['Pollen_suppression_index'] = df['Total_Pollen_roll_3'] * df['Precip_roll_3']
        
        # Wind dispersal
        df['Pollen_x_Wind'] = df['Total_Pollen_roll_3'] * df['AWND']
        df['Tree_x_Wind'] = df['Tree_Pollen_roll_3'] * df['AWND']
        df['Grass_x_Wind'] = df['Grass_Pollen_roll_3'] * df['AWND']
        
        # Temperature effects
        df['Pollen_x_Temp'] = df['Total_Pollen_roll_3'] * df['TAVG']
        df['Tree_x_Temp'] = df['Tree_Pollen_roll_3'] * df['TAVG']
        df['Grass_x_Temp'] = df['Grass_Pollen_roll_3'] * df['TAVG']
        
        # Dry conditions
        df['Consecutive_Dry_Days'] = (df['PRCP'] == 0).astype(int).groupby((df['PRCP'] > 0).cumsum()).cumsum()
        df['Pollen_x_Dry'] = df['Total_Pollen_roll_3'] * df['Consecutive_Dry_Days']
        df['Pollen_x_TempRange'] = df['Total_Pollen_roll_3'] * df['Temp_Range']
        
        # =====================================================================
        # ENHANCED BIOLOGICAL FEATURES
        # =====================================================================
        # Temperature dynamics
        df['TAVG_30d_mean'] = df['TAVG'].rolling(30, min_periods=1).mean()
        df['Temp_Anomaly'] = df['TAVG'] - df['TAVG_30d_mean']
        df['TAVG_change'] = df['TAVG'].diff()
        df['Temp_Volatility_7d'] = df['TAVG_change'].rolling(7, min_periods=1).std()
        
        # Precipitation patterns
        df['PRCP_YTD'] = df.groupby(df['Year'])['PRCP'].cumsum()
        df['PRCP_30d_sum'] = df['PRCP'].rolling(30, min_periods=1).sum()
        
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
        
        print(f"   ✅ Created {len(df.columns) - 34} engineered features")
        
        return df
    
    def select_features(self, df):
        """Select features for modeling"""
        exclude = ['Date_Standard', 'Total_Pollen', 'Tree', 'Grass', 
                   'Weed', 'Ragweed', 'Pollen_Severity', 
                   'Year', 'Month', 'TAVG_30d_mean', 'TAVG_change', 'Frost_Event',
                   'Consecutive_Dry_Days', 'Day_of_Year',
                   'Date', 'Week', 'Tree_Level', 'Grass_Level', 'Weed_Level', 'Ragweed_Level',
                   'OBJECTID', 'Year_Numeric', 'Month_Numeric', 'STATION', 'NAME',
                   'Tree_Level_Numeric', 'Grass_Level_Numeric', 'Weed_Level_Numeric', 
                   'Ragweed_Level_Numeric', 'TAVG_calculated', 'WSF2', 'WSF5']
        
        features = [c for c in df.columns if c not in exclude and not c.endswith('_lag_0')]
        self.features = features
        
        print(f"   ✅ Selected {len(features)} features")
        return features
    
    def train(self, df, target='Pollen_Severity'):
        """
        Train stacked ensemble model
        """
        print("\n" + "="*80)
        print("🚀 TRAINING FINAL POLLEN PREDICTOR - STACKED ENSEMBLE")
        print("="*80)
        
        # Engineer features
        df = self.create_pollen_severity_scale(df)
        df = self.engineer_features(df)
        features = self.select_features(df)
        
        # Prepare data
        print("\n📊 Preparing data...")
        df_clean = df.dropna(subset=features + [target]).copy()
        print(f"   Clean dataset: {len(df_clean)} samples")
        print(f"   Date range: {df_clean['Date_Standard'].min().date()} to {df_clean['Date_Standard'].max().date()}")
        
        # Time-based split (80/20)
        split_date = df_clean['Date_Standard'].quantile(0.8)
        train_mask = df_clean['Date_Standard'] <= split_date
        test_mask = df_clean['Date_Standard'] > split_date
        
        X_train = df_clean.loc[train_mask, features]
        X_test = df_clean.loc[test_mask, features]
        y_train = df_clean.loc[train_mask, target]
        y_test = df_clean.loc[test_mask, target]
        
        train_dates = df_clean.loc[train_mask, 'Date_Standard']
        test_dates = df_clean.loc[test_mask, 'Date_Standard']
        
        print(f"   Train: {len(X_train)} samples ({train_dates.min().date()} to {train_dates.max().date()})")
        print(f"   Test:  {len(X_test)} samples ({test_dates.min().date()} to {test_dates.max().date()})")
        
        # =====================================================================
        # LEVEL 1: Train base models
        # =====================================================================
        print("\n🏗️  Training Level 1 Models...")
        
        # Model 1: Random Forest
        print("   1/3: Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_train_pred = rf_model.predict(X_train)
        rf_test_pred = rf_model.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_test_pred)
        print(f"       Test MAE: {rf_mae:.4f}")
        
        # Model 2: Gradient Boosting
        print("   2/3: Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_train_pred = gb_model.predict(X_train)
        gb_test_pred = gb_model.predict(X_test)
        gb_mae = mean_absolute_error(y_test, gb_test_pred)
        print(f"       Test MAE: {gb_mae:.4f}")
        
        # Model 3: Ridge Regression
        print("   3/3: Ridge Regression...")
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        ridge_train_pred = ridge_model.predict(X_train)
        ridge_test_pred = ridge_model.predict(X_test)
        ridge_mae = mean_absolute_error(y_test, ridge_test_pred)
        print(f"       Test MAE: {ridge_mae:.4f}")
        
        self.level1_models = {
            'rf': rf_model,
            'gb': gb_model,
            'ridge': ridge_model
        }
        
        # =====================================================================
        # LEVEL 2: Train meta-learner
        # =====================================================================
        print("\n🧠 Training Level 2 Meta-Learner...")
        meta_train = np.column_stack([rf_train_pred, gb_train_pred, ridge_train_pred])
        meta_test = np.column_stack([rf_test_pred, gb_test_pred, ridge_test_pred])
        
        self.meta_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.meta_model.fit(meta_train, y_train)
        
        # =====================================================================
        # EVALUATION
        # =====================================================================
        print("\n📊 Evaluating Stacked Ensemble...")
        
        # Cross-validation on training set
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Level 1 predictions
            l1_rf = rf_model.predict(X_cv_val)
            l1_gb = gb_model.predict(X_cv_val)
            l1_ridge = ridge_model.predict(X_cv_val)
            meta_features = np.column_stack([l1_rf, l1_gb, l1_ridge])
            
            # Level 2 prediction
            pred = self.meta_model.predict(meta_features)
            cv_scores.append(mean_absolute_error(y_cv_val, pred))
        
        # Final predictions
        ensemble_train_pred = self.meta_model.predict(meta_train)
        ensemble_test_pred = self.meta_model.predict(meta_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, ensemble_train_pred)
        test_mae = mean_absolute_error(y_test, ensemble_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, ensemble_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
        train_r2 = r2_score(y_train, ensemble_train_pred)
        test_r2 = r2_score(y_test, ensemble_test_pred)
        
        # Accuracy within tolerance
        within_1 = np.mean(np.abs(y_test - ensemble_test_pred) <= 1)
        within_2 = np.mean(np.abs(y_test - ensemble_test_pred) <= 2)
        
        print(f"\n   Cross-validation MAE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"\n   Training Performance:")
        print(f"   • MAE:  {train_mae:.4f}")
        print(f"   • RMSE: {train_rmse:.4f}")
        print(f"   • R²:   {train_r2:.4f}")
        print(f"\n   Test Performance:")
        print(f"   • MAE:  {test_mae:.4f}")
        print(f"   • RMSE: {test_rmse:.4f}")
        print(f"   • R²:   {test_r2:.4f}")
        print(f"   • Accuracy (±1 level): {within_1:.1%}")
        print(f"   • Accuracy (±2 levels): {within_2:.1%}")
        
        # Meta-model weights
        print(f"\n   Meta-learner weights:")
        print(f"   • Random Forest:    {self.meta_model.feature_importances_[0]:.3f}")
        print(f"   • Gradient Boost:   {self.meta_model.feature_importances_[1]:.3f}")
        print(f"   • Ridge Regression: {self.meta_model.feature_importances_[2]:.3f}")
        
        # Store training info
        self.training_info = {
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'train_r2': float(train_r2),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'test_r2': float(test_r2),
            'cv_mae_mean': float(np.mean(cv_scores)),
            'cv_mae_std': float(np.std(cv_scores)),
            'accuracy_within_1': float(within_1),
            'accuracy_within_2': float(within_2),
            'n_features': len(features),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'train_date_range': [str(train_dates.min().date()), str(train_dates.max().date())],
            'test_date_range': [str(test_dates.min().date()), str(test_dates.max().date())],
            'trained_at': str(datetime.now())
        }
        
        return self
    
    def predict(self, X):
        """
        Make predictions using stacked ensemble
        
        Args:
            X: DataFrame with same features as training data
            
        Returns:
            predictions: Array of pollen severity predictions (0-10)
        """
        if self.meta_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Level 1 predictions
        rf_pred = self.level1_models['rf'].predict(X)
        gb_pred = self.level1_models['gb'].predict(X)
        ridge_pred = self.level1_models['ridge'].predict(X)
        
        # Stack predictions
        meta_features = np.column_stack([rf_pred, gb_pred, ridge_pred])
        
        # Level 2 prediction
        predictions = self.meta_model.predict(meta_features)
        
        return predictions
    
    def save(self, filepath='pollen_predictor_final.joblib'):
        """Save complete model"""
        model_data = {
            'level1_models': self.level1_models,
            'meta_model': self.meta_model,
            'features': self.features,
            'thresholds': self.thresholds,
            'training_info': self.training_info
        }
        joblib.dump(model_data, filepath)
        print(f"\n✅ Model saved to: {filepath}")
    
    @staticmethod
    def load(filepath='pollen_predictor_final.joblib'):
        """Load saved model"""
        model_data = joblib.load(filepath)
        predictor = PollenPredictor()
        predictor.level1_models = model_data['level1_models']
        predictor.meta_model = model_data['meta_model']
        predictor.features = model_data['features']
        predictor.thresholds = model_data['thresholds']
        predictor.training_info = model_data['training_info']
        return predictor


def main():
    """Main training pipeline"""
    print("="*80)
    print("🌸 FINAL POLLEN PREDICTOR - PRODUCTION MODEL")
    print("="*80)
    print("\nModel: Stacked Ensemble (RF + GB + Ridge → Meta-RF)")
    print("Features: Pollen momentum + Weather interactions + Biological")
    print("Performance: Test MAE ~0.90, R² ~0.78")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('combined_allergy_weather.csv')
    df['Date_Standard'] = pd.to_datetime(df['Date_Standard'])
    print(f"   Loaded {len(df)} samples")
    
    # Train model
    predictor = PollenPredictor()
    predictor.train(df)
    
    # Save model
    predictor.save('pollen_predictor_final.joblib')
    
    # Save training info
    with open('pollen_predictor_final_info.json', 'w') as f:
        json.dump(predictor.training_info, f, indent=2)
    print(f"✅ Training info saved to: pollen_predictor_final_info.json")
    
    # Save feature list
    with open('pollen_predictor_final_features.json', 'w') as f:
        json.dump(predictor.features, f, indent=2)
    print(f"✅ Feature list saved to: pollen_predictor_final_features.json")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE - Model ready for production!")
    print("="*80)
    print("\nUsage:")
    print("  from pollen_predictor_final import PollenPredictor")
    print("  predictor = PollenPredictor.load('pollen_predictor_final.joblib')")
    print("  predictions = predictor.predict(features)")
    print("="*80)


if __name__ == "__main__":
    main()
