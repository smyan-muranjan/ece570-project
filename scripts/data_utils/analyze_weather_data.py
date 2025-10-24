
"""
Weather Data Analysis and Aggregation Strategy
Analyzes weather data completeness to determine the best aggregation approach
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_weather_data(file_path: str):
    """Analyze weather data structure and completeness"""
    print("🌤️ Analyzing Weather Dataset...")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    
    # Show columns
    print(f"\nColumns: {list(df.columns)}")
    
    # Analyze stations
    station_counts = df['STATION'].value_counts()
    print(f"\n📊 Station Analysis:")
    print(f"Total stations: {len(station_counts)}")
    print(f"Records per station (top 10):")
    print(station_counts.head(10))
    
    # Check target station
    target_station = "USW00014742"
    target_data = df[df['STATION'] == target_station]
    print(f"\n🎯 Target Station ({target_station}):")
    if len(target_data) > 0:
        print(f"Station name: {target_data['NAME'].iloc[0]}")
        print(f"Records: {len(target_data)}")
        print(f"Date range: {target_data['DATE'].min()} to {target_data['DATE'].max()}")
    else:
        print("Station not found!")
        return None, None
    
    return df, target_data

def analyze_data_completeness(df, target_data):
    """Analyze data completeness by variable and station"""
    print("\n📈 Data Completeness Analysis...")
    
    # Key weather variables for allergy prediction
    weather_vars = ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND', 'WSF2', 'WSF5']
    
    print("\n🌡️ Target Station Completeness:")
    target_completeness = {}
    for var in weather_vars:
        if var in target_data.columns:
            non_null = target_data[var].notna().sum()
            total = len(target_data)
            pct = (non_null / total) * 100
            target_completeness[var] = pct
            print(f"{var}: {non_null}/{total} ({pct:.1f}%)")
    
    print("\n🌍 All Stations Completeness (by date):")
    # Group by date and check completeness
    daily_completeness = {}
    for var in weather_vars:
        if var in df.columns:
            daily_stats = df.groupby('DATE')[var].agg(['count', 'size']).reset_index()
            daily_stats['completeness'] = (daily_stats['count'] / daily_stats['size']) * 100
            avg_completeness = daily_stats['completeness'].mean()
            daily_completeness[var] = avg_completeness
            print(f"{var}: {avg_completeness:.1f}% average daily completeness")
    
    return target_completeness, daily_completeness, weather_vars

def analyze_geographic_spread(df):
    """Analyze geographic spread of stations"""
    print("\n🗺️ Geographic Analysis...")
    
    # Get unique stations with their names
    stations = df[['STATION', 'NAME']].drop_duplicates()
    print(f"Unique stations: {len(stations)}")
    
    # Look for Vermont/Burlington area stations
    vt_stations = stations[stations['NAME'].str.contains('VT|VERMONT|BURLINGTON', case=False, na=False)]
    print(f"\nVermont/Burlington area stations: {len(vt_stations)}")
    for _, row in vt_stations.head(10).iterrows():
        station_data = df[df['STATION'] == row['STATION']]
        print(f"  {row['STATION']}: {row['NAME']} ({len(station_data)} records)")

def recommend_aggregation_strategy(target_completeness, daily_completeness, weather_vars):
    """Recommend the best aggregation strategy"""
    print("\n🎯 AGGREGATION STRATEGY RECOMMENDATION")
    print("=" * 60)
    
    # Compare completeness
    print("\nCompleteness Comparison:")
    print("Variable | Target Station | All Stations | Recommendation")
    print("-" * 60)
    
    recommendations = {}
    
    for var in weather_vars:
        target_pct = target_completeness.get(var, 0)
        all_pct = daily_completeness.get(var, 0)
        
        if target_pct >= 80:
            rec = "✅ Use Target Station"
            strategy = "single"
        elif all_pct >= 80:
            rec = "🔄 Aggregate All Stations"
            strategy = "aggregate"
        elif target_pct > all_pct:
            rec = "⚠️ Use Target (Limited)"
            strategy = "single_limited"
        else:
            rec = "⚠️ Aggregate (Limited)"
            strategy = "aggregate_limited"
        
        recommendations[var] = strategy
        print(f"{var:8} | {target_pct:13.1f}% | {all_pct:11.1f}% | {rec}")
    
    # Overall recommendation
    single_count = sum(1 for s in recommendations.values() if 'single' in s)
    aggregate_count = sum(1 for s in recommendations.values() if 'aggregate' in s)
    
    print(f"\n📊 Overall Strategy:")
    if single_count > aggregate_count:
        print("🎯 RECOMMENDED: Use Target Station (USW00014742)")
        print("   Reasons:")
        print("   • Higher completeness for most variables")
        print("   • Consistent single location")
        print("   • Cleaner data with fewer missing values")
        overall_strategy = "single"
    else:
        print("🔄 RECOMMENDED: Aggregate Multiple Stations")
        print("   Reasons:")
        print("   • Better overall completeness")
        print("   • More robust to individual station failures")
        print("   • Regional averaging reduces noise")
        overall_strategy = "aggregate"
    
    return overall_strategy, recommendations

def create_sample_aggregation(df, strategy="single"):
    """Create sample aggregated data"""
    print(f"\n🔧 Creating Sample Aggregation (Strategy: {strategy})...")
    
    weather_vars = ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND']
    
    if strategy == "single":
        # Filter to target station only
        target_station = "USW00014742"
        filtered_df = df[df['STATION'] == target_station].copy()
        
        # Basic cleaning
        result = filtered_df[['DATE'] + weather_vars].copy()
        
    else:  # aggregate
        # Group by date and aggregate across all stations
        agg_functions = {
            'TMAX': 'mean',    # Average temperature across stations
            'TMIN': 'mean',    # Average temperature across stations  
            'TAVG': 'mean',    # Average temperature across stations
            'PRCP': 'mean',    # Average precipitation
            'AWND': 'mean'     # Average wind speed
        }
        
        # Filter to available columns
        available_vars = [var for var in weather_vars if var in df.columns]
        available_agg = {k: v for k, v in agg_functions.items() if k in available_vars}
        
        result = df.groupby('DATE')[available_vars].agg(available_agg).reset_index()
    
    # Convert temperature from Fahrenheit to Celsius (if needed)
    temp_cols = ['TMAX', 'TMIN', 'TAVG']
    for col in temp_cols:
        if col in result.columns:
            # Check if values look like Fahrenheit (typically > 50)
            if result[col].mean() > 50:
                result[col] = (result[col] - 32) * 5/9
    
    # Convert wind speed from mph to m/s (if needed)
    if 'AWND' in result.columns:
        # AWND is typically in m/s already, but check
        if result['AWND'].mean() > 10:  # Likely mph
            result['AWND'] = result['AWND'] * 0.44704
    
    print(f"Sample aggregated data shape: {result.shape}")
    print(f"Date range: {result['DATE'].min()} to {result['DATE'].max()}")
    print("\nFirst 5 rows:")
    print(result.head())
    
    # Check data completeness
    print(f"\nData completeness:")
    for col in result.columns:
        if col != 'DATE':
            completeness = result[col].notna().mean() * 100
            print(f"{col}: {completeness:.1f}%")
    
    return result

def main():
    """Main analysis function"""
    print("🌤️ Weather Data Aggregation Strategy Analysis")
    print("=" * 60)
    
    # Analyze data
    df, target_data = analyze_weather_data('data/weather_data.csv')
    
    if df is None:
        return
    
    # Completeness analysis
    target_completeness, daily_completeness, weather_vars = analyze_data_completeness(df, target_data)
    
    # Geographic analysis
    analyze_geographic_spread(df)
    
    # Get recommendation
    strategy, var_recommendations = recommend_aggregation_strategy(
        target_completeness, daily_completeness, weather_vars
    )
    
    # Create sample
    sample_data = create_sample_aggregation(df, strategy)
    
    print(f"\n✅ Analysis Complete!")
    print(f"📄 Recommended approach: {strategy}")
    print(f"🎯 Next step: Implement the recommended strategy and join with allergy data")

if __name__ == "__main__":
    main()