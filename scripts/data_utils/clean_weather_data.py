
"""
Weather Data Cleaning Script
Filters weather data to Burlington International Airport (USW00014742) only
Preserves original dataset and creates cleaned version
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_and_verify_original(file_path: str):
    """Load original weather data and verify structure"""
    print("📂 Loading original weather dataset...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Original file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    print(f"✅ Original dataset loaded successfully")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    print(f"   Total stations: {df['STATION'].nunique()}")
    print(f"   Total records: {len(df)}")
    
    return df

def filter_burlington_airport(df: pd.DataFrame):
    """Filter data to Burlington International Airport only"""
    print("\n🛫 Filtering to Burlington International Airport...")
    
    target_station = "USW00014742"
    
    # Check if station exists
    if target_station not in df['STATION'].values:
        raise ValueError(f"Target station {target_station} not found in dataset!")
    
    # Filter to target station
    filtered_df = df[df['STATION'] == target_station].copy()
    
    # Get station info
    station_name = filtered_df['NAME'].iloc[0]
    
    print(f"✅ Filtered to target station:")
    print(f"   Station ID: {target_station}")
    print(f"   Station Name: {station_name}")
    print(f"   Records: {len(filtered_df)}")
    print(f"   Date range: {filtered_df['DATE'].min()} to {filtered_df['DATE'].max()}")
    print(f"   Data reduction: {len(df)} → {len(filtered_df)} records ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return filtered_df

def clean_weather_variables(df: pd.DataFrame):
    """Clean and standardize weather variables"""
    print("\n🧹 Cleaning weather variables...")
    
    # Create a copy for cleaning
    cleaned_df = df.copy()
    
    # Key weather variables for allergy prediction
    weather_vars = ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND', 'WSF2', 'WSF5']
    
    print("📊 Data completeness before cleaning:")
    initial_completeness = {}
    for var in weather_vars:
        if var in cleaned_df.columns:
            completeness = cleaned_df[var].notna().mean() * 100
            initial_completeness[var] = completeness
            print(f"   {var}: {completeness:.1f}%")
    
    # Convert date to datetime for easier handling
    cleaned_df['DATE'] = pd.to_datetime(cleaned_df['DATE'])
    
    # Temperature unit conversion (Fahrenheit to Celsius if needed)
    temp_vars = ['TMAX', 'TMIN', 'TAVG']
    for var in temp_vars:
        if var in cleaned_df.columns:
            # Check if values look like Fahrenheit (Burlington temps should not average > 30°C)
            non_null_temps = cleaned_df[var].dropna()
            if len(non_null_temps) > 0 and non_null_temps.mean() > 30:
                print(f"   Converting {var} from Fahrenheit to Celsius")
                cleaned_df[var] = (cleaned_df[var] - 32) * 5.0 / 9.0
            else:
                print(f"   {var} appears to already be in Celsius")
    
    # Handle missing TAVG by calculating from TMAX and TMIN
    if 'TAVG' in cleaned_df.columns and 'TMAX' in cleaned_df.columns and 'TMIN' in cleaned_df.columns:
        missing_tavg = cleaned_df['TAVG'].isna()
        has_tmax_tmin = cleaned_df['TMAX'].notna() & cleaned_df['TMIN'].notna()
        can_calculate = missing_tavg & has_tmax_tmin
        
        if can_calculate.sum() > 0:
            print(f"   Calculating missing TAVG from TMAX/TMIN for {can_calculate.sum()} records")
            cleaned_df.loc[can_calculate, 'TAVG'] = (
                cleaned_df.loc[can_calculate, 'TMAX'] + cleaned_df.loc[can_calculate, 'TMIN']
            ) / 2
            
            # Add flag to indicate calculated values
            cleaned_df['TAVG_calculated'] = can_calculate
    
    # Wind speed conversion (mph to m/s if needed)
    wind_vars = ['AWND', 'WSF2', 'WSF5']
    for var in wind_vars:
        if var in cleaned_df.columns:
            non_null_wind = cleaned_df[var].dropna()
            if len(non_null_wind) > 0 and non_null_wind.mean() > 20:  # Likely mph
                print(f"   Converting {var} from mph to m/s")
                cleaned_df[var] = cleaned_df[var] * 0.44704
            else:
                print(f"   {var} appears to already be in m/s")
    
    # Precipitation is typically in inches, convert to mm
    if 'PRCP' in cleaned_df.columns:
        non_null_prcp = cleaned_df['PRCP'].dropna()
        if len(non_null_prcp) > 0 and non_null_prcp.max() < 20:  # Likely inches
            print(f"   Converting PRCP from inches to mm")
            cleaned_df['PRCP'] = cleaned_df['PRCP'] * 25.4
        else:
            print(f"   PRCP appears to already be in mm")
    
    print("\n📊 Data completeness after cleaning:")
    final_completeness = {}
    for var in weather_vars:
        if var in cleaned_df.columns:
            completeness = cleaned_df[var].notna().mean() * 100
            final_completeness[var] = completeness
            improvement = completeness - initial_completeness.get(var, 0)
            if improvement > 0:
                print(f"   {var}: {completeness:.1f}% (+{improvement:.1f}%)")
            else:
                print(f"   {var}: {completeness:.1f}%")
    
    return cleaned_df

def select_key_variables(df: pd.DataFrame):
    """Select and order key variables for analysis"""
    print("\n📋 Selecting key variables...")
    
    # Essential variables for allergy prediction
    key_vars = [
        'STATION',
        'NAME', 
        'DATE',
        'TMAX',      # Maximum temperature
        'TMIN',      # Minimum temperature  
        'TAVG',      # Average temperature
        'PRCP',      # Precipitation
        'AWND',      # Average wind speed
        'WSF2',      # Fastest 2-minute wind speed
        'WSF5',      # Fastest 5-second wind speed
    ]
    
    # Add calculated flag if it exists
    if 'TAVG_calculated' in df.columns:
        key_vars.append('TAVG_calculated')
    
    # Filter to available columns
    available_vars = [var for var in key_vars if var in df.columns]
    result_df = df[available_vars].copy()
    
    print(f"   Selected {len(available_vars)} variables: {available_vars}")
    print(f"   Final dataset shape: {result_df.shape}")
    
    return result_df

def save_cleaned_data(df: pd.DataFrame, output_file: str = 'weather_cleaned.csv'):
    """Save cleaned data and verify integrity"""
    print(f"\n💾 Saving cleaned data to {output_file}...")
    
    # Sort by date for consistency
    df_sorted = df.sort_values('DATE').reset_index(drop=True)
    
    # Save to CSV
    df_sorted.to_csv(output_file, index=False)
    
    print(f"✅ File saved successfully!")
    print(f"   Output file: {output_file}")
    print(f"   File size: {os.path.getsize(output_file)} bytes")
    
    return output_file

def verify_cleaned_data(original_file: str, cleaned_file: str):
    """Verify the cleaned data integrity"""
    print(f"\n🔍 Verifying cleaned data integrity...")
    
    # Load both files
    original_df = pd.read_csv(original_file)
    cleaned_df = pd.read_csv(cleaned_file)
    
    # Check that original is preserved
    print(f"✅ Original file preserved:")
    print(f"   Original records: {len(original_df)}")
    print(f"   Original stations: {original_df['STATION'].nunique()}")
    
    # Check cleaned data
    target_station = "USW00014742"
    original_target = original_df[original_df['STATION'] == target_station]
    
    print(f"✅ Cleaned data verification:")
    print(f"   Cleaned records: {len(cleaned_df)}")
    print(f"   Expected records: {len(original_target)}")
    print(f"   Records match: {len(cleaned_df) == len(original_target)}")
    
    # Date range check
    print(f"   Date range: {cleaned_df['DATE'].min()} to {cleaned_df['DATE'].max()}")
    
    # Data quality check
    key_vars = ['TMAX', 'TMIN', 'PRCP', 'AWND']
    print(f"📊 Key variable completeness:")
    for var in key_vars:
        if var in cleaned_df.columns:
            completeness = cleaned_df[var].notna().mean() * 100
            print(f"   {var}: {completeness:.1f}%")
    
    # Sample data check
    print(f"\n📋 Sample of cleaned data (first 3 rows):")
    print(cleaned_df.head(3).to_string())
    
    return True

def main():
    """Main processing function"""
    print("🌤️ Weather Data Cleaning for Burlington International Airport")
    print("=" * 70)
    
    # File paths
    original_file = 'data/weather_data.csv'
    output_file = 'data/weather_cleaned.csv'
    
    try:
        # Step 1: Load original data
        original_df = load_and_verify_original(original_file)
        
        # Step 2: Filter to Burlington Airport
        filtered_df = filter_burlington_airport(original_df)
        
        # Step 3: Clean weather variables
        cleaned_df = clean_weather_variables(filtered_df)
        
        # Step 4: Select key variables
        final_df = select_key_variables(cleaned_df)
        
        # Step 5: Save cleaned data
        output_path = save_cleaned_data(final_df, output_file)
        
        # Step 6: Verify results
        verify_cleaned_data(original_file, output_file)
        
        print("\n" + "=" * 70)
        print("✅ CLEANING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📂 Original file preserved: {original_file}")
        print(f"📄 Cleaned file created: {output_file}")
        print(f"🎯 Ready for merging with allergy data!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())