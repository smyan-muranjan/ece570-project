#!/usr/bin/env python3
"""
JSON to CSV Converter for Allergy/Pollen Data
Converts ESRI JSON format allergy data to CSV for ML analysis
"""

import json
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file and extract features
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[Dict]: List of feature attributes
    """
    print(f"Loading data from {file_path}...")
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the features from the ESRI JSON structure
    features = data.get('features', [])
    records = []
    
    for feature in features:
        attributes = feature.get('attributes', {})
        records.append(attributes)
    
    print(f"Loaded {len(records)} records from {file_path}")
    return records

def standardize_date_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize date formats for better consistency
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with standardized dates
    """
    print("Standardizing date formats...")
    
    # Convert the Date column to datetime and create a standardized date column
    df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date_Standard'] = df['Date_Parsed'].dt.strftime('%Y-%m-%d')
    
    # Add some useful date features for ML
    df['Year_Numeric'] = df['Date_Parsed'].dt.year
    df['Month_Numeric'] = df['Date_Parsed'].dt.month
    df['Day_of_Year'] = df['Date_Parsed'].dt.dayofyear
    df['Day_of_Week'] = df['Date_Parsed'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    return df

def clean_and_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process the dataframe for ML readiness
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Cleaning and processing data...")
    
    # Convert categorical levels to numeric values for ML
    level_mapping = {
        'Absent': 0,
        'Low': 1,
        'Moderate': 2,
        'High': 3,
        'Very High': 4,
        'Not Collected': None
    }
    
    # Apply mapping to level columns
    level_columns = ['Tree_Level', 'Grass_Level', 'Weed_Level', 'Ragweed_Level']
    for col in level_columns:
        if col in df.columns:
            df[f'{col}_Numeric'] = df[col].map(level_mapping)
    
    # Handle null values in numeric columns
    numeric_columns = ['Tree', 'Grass', 'Weed', 'Ragweed', 'Total_Pollen']
    for col in numeric_columns:
        if col in df.columns:
            # Fill null values with 0 for pollen counts (assuming 0 means no pollen detected)
            df[col] = df[col].fillna(0)
    
    # Remove the parsed date column as we have the standard one
    if 'Date_Parsed' in df.columns:
        df = df.drop('Date_Parsed', axis=1)
    
    # Sort by date for time series analysis
    if 'Date_Standard' in df.columns:
        df = df.sort_values('Date_Standard').reset_index(drop=True)
    
    return df

def convert_json_to_csv(json_files: List[str], output_file: str = 'combined_allergy_data.csv') -> str:
    """
    Convert multiple JSON files to a single CSV file
    
    Args:
        json_files (List[str]): List of JSON file paths
        output_file (str): Output CSV file name
        
    Returns:
        str: Path to the created CSV file
    """
    all_records = []
    
    # Load data from all JSON files
    for json_file in json_files:
        if os.path.exists(json_file):
            records = load_json_data(json_file)
            all_records.extend(records)
        else:
            print(f"Warning: File {json_file} not found!")
    
    if not all_records:
        raise ValueError("No data found in any of the JSON files!")
    
    # Convert to DataFrame
    print(f"Converting {len(all_records)} total records to DataFrame...")
    df = pd.DataFrame(all_records)
    
    # Process the data
    df = standardize_date_format(df)
    df = clean_and_process_data(df)
    
    # Remove duplicates based on date (in case there are overlapping records)
    if 'Date_Standard' in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Date_Standard'], keep='last')
        final_count = len(df)
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} duplicate records")
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(json_files[0]), output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\nConversion completed!")
    print(f"Output file: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date_Standard'].min()} to {df['Date_Standard'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic statistics
    print("\nBasic Statistics:")
    numeric_cols = ['Tree', 'Grass', 'Weed', 'Ragweed', 'Total_Pollen']
    for col in numeric_cols:
        if col in df.columns:
            print(f"{col}: mean={df[col].mean():.2f}, max={df[col].max():.2f}, min={df[col].min():.2f}")
    
    return output_path

def main():
    """
    Main function to run the conversion
    """
    # Define input files
    json_files = [
        'query.json',
        'query2.json'
    ]
    
    try:
        # Convert to CSV
        csv_file = convert_json_to_csv(json_files, 'allergy_pollen_data.csv')
        
        print(f"\n✅ Successfully converted JSON files to CSV!")
        print(f"📄 Output file: {csv_file}")
        print("\n📊 The CSV file is now ready for:")
        print("   • Weather data integration")
        print("   • Machine learning model training")
        print("   • Time series analysis")
        print("   • Data visualization")
        
        # Show first few rows
        df = pd.read_csv(csv_file)
        print(f"\n📋 Preview of first 5 rows:")
        print(df.head().to_string())
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())