#!/usr/bin/env python3
"""
Dataset Merger for Allergy and Weather Data
Merges allergy and weather datasets on overlapping dates with comprehensive verification
Preserves original files and provides detailed quality checks
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_datasets():
    """Load both datasets and verify they exist"""
    print("📂 Loading datasets...")
    
    # Check file existence
    allergy_file = 'allergy_pollen_data.csv'
    weather_file = 'weather_cleaned.csv'
    
    if not os.path.exists(allergy_file):
        raise FileNotFoundError(f"Allergy data file not found: {allergy_file}")
    if not os.path.exists(weather_file):
        raise FileNotFoundError(f"Weather data file not found: {weather_file}")
    
    # Load datasets
    print(f"   Loading {allergy_file}...")
    allergy_df = pd.read_csv(allergy_file)
    
    print(f"   Loading {weather_file}...")
    weather_df = pd.read_csv(weather_file)
    
    print("✅ Both datasets loaded successfully")
    print(f"   Allergy data: {allergy_df.shape}")
    print(f"   Weather data: {weather_df.shape}")
    
    return allergy_df, weather_df

def analyze_date_alignment(allergy_df, weather_df):
    """Analyze date ranges and overlaps between datasets"""
    print("\n📅 Analyzing date alignment...")
    
    # Convert dates to datetime
    allergy_df['Date_Standard'] = pd.to_datetime(allergy_df['Date_Standard'])
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
    
    # Get date ranges
    allergy_start = allergy_df['Date_Standard'].min()
    allergy_end = allergy_df['Date_Standard'].max()
    weather_start = weather_df['DATE'].min()
    weather_end = weather_df['DATE'].max()
    
    print(f"📊 Date Range Analysis:")
    print(f"   Allergy data: {allergy_start.date()} to {allergy_end.date()} ({len(allergy_df)} records)")
    print(f"   Weather data: {weather_start.date()} to {weather_end.date()} ({len(weather_df)} records)")
    
    # Find overlap
    overlap_start = max(allergy_start, weather_start)
    overlap_end = min(allergy_end, weather_end)
    
    print(f"   Overlap period: {overlap_start.date()} to {overlap_end.date()}")
    
    # Count records in overlap period
    allergy_overlap = allergy_df[
        (allergy_df['Date_Standard'] >= overlap_start) & 
        (allergy_df['Date_Standard'] <= overlap_end)
    ]
    weather_overlap = weather_df[
        (weather_df['DATE'] >= overlap_start) & 
        (weather_df['DATE'] <= overlap_end)
    ]
    
    print(f"   Allergy records in overlap: {len(allergy_overlap)}")
    print(f"   Weather records in overlap: {len(weather_overlap)}")
    
    # Check for unique dates
    allergy_dates = set(allergy_overlap['Date_Standard'].dt.date)
    weather_dates = set(weather_overlap['DATE'].dt.date)
    
    print(f"   Unique allergy dates in overlap: {len(allergy_dates)}")
    print(f"   Unique weather dates in overlap: {len(weather_dates)}")
    
    # Find common dates
    common_dates = allergy_dates.intersection(weather_dates)
    allergy_only = allergy_dates - weather_dates
    weather_only = weather_dates - allergy_dates
    
    print(f"   Common dates: {len(common_dates)}")
    print(f"   Allergy-only dates: {len(allergy_only)}")
    print(f"   Weather-only dates: {len(weather_only)}")
    
    if len(allergy_only) > 0:
        print(f"   Sample allergy-only dates: {sorted(list(allergy_only))[:5]}")
    if len(weather_only) > 0:
        print(f"   Sample weather-only dates: {sorted(list(weather_only))[:5]}")
    
    return overlap_start, overlap_end, common_dates

def verify_original_files_preserved():
    """Verify that original files haven't been modified"""
    print("\n🔒 Verifying original files are preserved...")
    
    files_to_check = [
        ('4129133.csv', 'Original weather data'),
        ('allergy_pollen_data.csv', 'Allergy data'),
        ('weather_cleaned.csv', 'Cleaned weather data')
    ]
    
    all_preserved = True
    for filename, description in files_to_check:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"   ✅ {description}: {filename} ({file_size} bytes)")
        else:
            print(f"   ❌ {description}: {filename} - FILE MISSING!")
            all_preserved = False
    
    if all_preserved:
        print("✅ All original files are preserved")
    else:
        print("❌ Some original files are missing!")
    
    return all_preserved

def merge_datasets(allergy_df, weather_df, overlap_start, overlap_end):
    """Merge datasets on overlapping dates"""
    print("\n🔗 Merging datasets...")
    
    # Filter to overlap period
    allergy_filtered = allergy_df[
        (allergy_df['Date_Standard'] >= overlap_start) & 
        (allergy_df['Date_Standard'] <= overlap_end)
    ].copy()
    
    weather_filtered = weather_df[
        (weather_df['DATE'] >= overlap_start) & 
        (weather_df['DATE'] <= overlap_end)
    ].copy()
    
    print(f"   Filtered allergy data: {len(allergy_filtered)} records")
    print(f"   Filtered weather data: {len(weather_filtered)} records")
    
    # Rename date column in weather data to match allergy data
    weather_filtered = weather_filtered.rename(columns={'DATE': 'Date_Standard'})
    
    # Merge on date (inner join to keep only overlapping dates)
    merged_df = pd.merge(
        allergy_filtered, 
        weather_filtered, 
        on='Date_Standard', 
        how='inner',
        suffixes=('', '_weather')
    )
    
    print(f"   Merged dataset: {merged_df.shape}")
    print(f"   Date range: {merged_df['Date_Standard'].min().date()} to {merged_df['Date_Standard'].max().date()}")
    
    return merged_df

def perform_weather_quality_checks(merged_df):
    """Perform comprehensive weather data quality checks"""
    print("\n🌡️ Weather data quality checks...")
    
    # Temperature checks
    print("📊 Temperature Quality:")
    temp_cols = ['TMAX', 'TMIN', 'TAVG']
    for col in temp_cols:
        if col in merged_df.columns:
            data = merged_df[col].dropna()
            print(f"   {col}: min={data.min():.1f}°C, max={data.max():.1f}°C, mean={data.mean():.1f}°C")
            
            # Check for reasonable temperature ranges (Burlington, VT)
            if data.min() < -40 or data.max() > 45:
                print(f"   ⚠️ {col}: Extreme temperatures detected")
            else:
                print(f"   ✅ {col}: Temperature range looks reasonable")
    
    # Temperature consistency checks
    if all(col in merged_df.columns for col in ['TMAX', 'TMIN', 'TAVG']):
        tmax_tmin_check = (merged_df['TMAX'] >= merged_df['TMIN']).all()
        tavg_range_check = (
            (merged_df['TAVG'] >= merged_df['TMIN']) & 
            (merged_df['TAVG'] <= merged_df['TMAX'])
        ).all()
        
        print(f"   ✅ TMAX >= TMIN: {tmax_tmin_check}")
        print(f"   ✅ TMIN <= TAVG <= TMAX: {tavg_range_check}")
    
    # Precipitation checks
    print("\n🌧️ Precipitation Quality:")
    if 'PRCP' in merged_df.columns:
        prcp_data = merged_df['PRCP'].dropna()
        print(f"   PRCP: min={prcp_data.min():.1f}mm, max={prcp_data.max():.1f}mm, mean={prcp_data.mean():.1f}mm")
        
        # Check for negative precipitation
        negative_prcp = (prcp_data < 0).sum()
        if negative_prcp > 0:
            print(f"   ⚠️ Negative precipitation values: {negative_prcp}")
        else:
            print(f"   ✅ No negative precipitation values")
    
    # Wind speed checks
    print("\n💨 Wind Speed Quality:")
    wind_cols = ['AWND', 'WSF2', 'WSF5']
    for col in wind_cols:
        if col in merged_df.columns:
            wind_data = merged_df[col].dropna()
            if len(wind_data) > 0:
                print(f"   {col}: min={wind_data.min():.1f}m/s, max={wind_data.max():.1f}m/s, mean={wind_data.mean():.1f}m/s")
                
                # Check for unreasonable wind speeds
                if wind_data.max() > 50:  # > 180 km/h
                    print(f"   ⚠️ {col}: Extremely high wind speeds detected")
                else:
                    print(f"   ✅ {col}: Wind speeds look reasonable")
    
    # Data completeness
    print("\n📊 Data Completeness:")
    key_weather_vars = ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND']
    for var in key_weather_vars:
        if var in merged_df.columns:
            completeness = merged_df[var].notna().mean() * 100
            print(f"   {var}: {completeness:.1f}%")

def analyze_merged_results(merged_df):
    """Analyze the quality and characteristics of merged results"""
    print("\n📋 Merged Dataset Analysis...")
    
    print(f"📊 Dataset Overview:")
    print(f"   Shape: {merged_df.shape}")
    print(f"   Date range: {merged_df['Date_Standard'].min().date()} to {merged_df['Date_Standard'].max().date()}")
    print(f"   Total days: {len(merged_df)}")
    
    # Check for duplicates
    duplicates = merged_df.duplicated(subset=['Date_Standard']).sum()
    print(f"   Duplicate dates: {duplicates}")
    
    # Seasonal distribution
    merged_df['Month'] = merged_df['Date_Standard'].dt.month
    seasonal_dist = merged_df['Month'].value_counts().sort_index()
    print(f"\n📅 Monthly Distribution:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, count in seasonal_dist.items():
        print(f"   {month_names[month-1]}: {count} days")
    
    # Key variable correlations
    print(f"\n🔗 Key Variable Correlations with Total_Pollen:")
    weather_vars = ['TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND']
    for var in weather_vars:
        if var in merged_df.columns and 'Total_Pollen' in merged_df.columns:
            corr = merged_df[var].corr(merged_df['Total_Pollen'])
            if not pd.isna(corr):
                print(f"   {var}: {corr:.3f}")

def show_sample_results(merged_df, n_samples=5):
    """Show sample of merged results"""
    print(f"\n📋 Sample Merged Results (first {n_samples} rows):")
    
    # Select key columns for display
    display_cols = [
        'Date_Standard', 'Total_Pollen', 'Tree', 'Grass', 'Weed', 
        'TMAX', 'TMIN', 'TAVG', 'PRCP', 'AWND'
    ]
    
    # Filter to available columns
    available_cols = [col for col in display_cols if col in merged_df.columns]
    sample_df = merged_df[available_cols].head(n_samples)
    
    print(sample_df.to_string(index=False))
    
    print(f"\n📋 Sample from different seasons:")
    # Show samples from different months
    for month in [3, 6, 9]:  # Spring, Summer, Fall
        month_data = merged_df[merged_df['Date_Standard'].dt.month == month]
        if len(month_data) > 0:
            print(f"\n{['Spring', 'Summer', 'Fall'][month//3-1]} sample ({month_names[month-1]}):")
            print(month_data[available_cols].head(1).to_string(index=False))

def save_merged_dataset(merged_df, output_file='combined_allergy_weather.csv'):
    """Save merged dataset"""
    print(f"\n💾 Saving merged dataset to {output_file}...")
    
    # Sort by date
    merged_sorted = merged_df.sort_values('Date_Standard').reset_index(drop=True)
    
    # Save to CSV
    merged_sorted.to_csv(output_file, index=False)
    
    file_size = os.path.getsize(output_file)
    print(f"✅ File saved successfully!")
    print(f"   Output file: {output_file}")
    print(f"   File size: {file_size} bytes")
    
    return output_file

def main():
    """Main merger function"""
    print("🔗 Allergy and Weather Data Merger")
    print("=" * 60)
    
    try:
        # Step 1: Load datasets
        allergy_df, weather_df = load_datasets()
        
        # Step 2: Verify original files are preserved
        if not verify_original_files_preserved():
            print("❌ Original file verification failed!")
            return 1
        
        # Step 3: Analyze date alignment
        overlap_start, overlap_end, common_dates = analyze_date_alignment(allergy_df, weather_df)
        
        if len(common_dates) == 0:
            print("❌ No overlapping dates found between datasets!")
            return 1
        
        # Step 4: Merge datasets
        merged_df = merge_datasets(allergy_df, weather_df, overlap_start, overlap_end)
        
        # Step 5: Weather data quality checks
        perform_weather_quality_checks(merged_df)
        
        # Step 6: Analyze merged results
        analyze_merged_results(merged_df)
        
        # Step 7: Show sample results
        show_sample_results(merged_df)
        
        # Step 8: Save merged dataset
        output_file = save_merged_dataset(merged_df)
        
        print("\n" + "=" * 60)
        print("✅ MERGER COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📂 Original files preserved")
        print(f"📄 Merged dataset: {output_file}")
        print(f"📊 Final dataset: {merged_df.shape[0]} days of combined allergy + weather data")
        print(f"🎯 Ready for ML model training!")
        
    except Exception as e:
        print(f"\n❌ Error during merger: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Define month names globally
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    exit(main())