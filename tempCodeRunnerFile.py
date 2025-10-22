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