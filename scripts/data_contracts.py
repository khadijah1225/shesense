"""
Data contracts and utilities for SheSense project.

Standardizes column names across datasets and provides resampling/alignment helpers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Standard column schema for SheSense datasets
SHESENSE_SCHEMA = {
    'timestamp': 'datetime64[ns]',
    'subject': 'str',
    'hr': 'float64',
    'rmssd': 'float64',
    'sdnn': 'float64', 
    'eda_scl': 'float64',
    'eda_peak_rate': 'float64',
    'temp': 'float64',
    'acc_mag': 'float64',
    'label': 'str'
}

# Column mapping for different datasets
HONGN_COLUMN_MAP = {
    'timestamp': 'timestamp',
    'subject_id': 'subject',
    'BVP_HR': 'hr',
    'EDA': 'eda_scl',
    'TEMP': 'temp',
    'ACC_magnitude': 'acc_mag',
    'activity_label': 'label'
}

SWELL_COLUMN_MAP = {
    'timestamp': 'timestamp',
    'participant_id': 'subject',
    'HR': 'hr',
    'RMSSD': 'rmssd',
    'SDNN': 'sdnn',
    'EDA_tonic': 'eda_scl',
    'stress_label': 'label'
}

WESAD_COLUMN_MAP = {
    'timestamp': 'timestamp',
    'hr': 'hr',
    'temp': 'temp',
    'label': 'label'
}

def harmonize_columns(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """
    Harmonize column names to SheSense standard schema.
    
    Args:
        df: Input dataframe
        dataset_type: One of 'hongn', 'swell', 'wesad'
    
    Returns:
        DataFrame with standardized column names
    """
    if dataset_type.lower() == 'hongn':
        column_map = HONGN_COLUMN_MAP
    elif dataset_type.lower() == 'swell':
        column_map = SWELL_COLUMN_MAP
    elif dataset_type.lower() == 'wesad':
        column_map = WESAD_COLUMN_MAP
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Only rename columns that exist in the dataframe
    available_map = {old: new for old, new in column_map.items() if old in df.columns}
    df_harmonized = df.rename(columns=available_map)
    
    # Ensure timestamp is datetime
    if 'timestamp' in df_harmonized.columns:
        df_harmonized['timestamp'] = pd.to_datetime(df_harmonized['timestamp'])
    
    # Add missing columns with NaN
    for col in SHESENSE_SCHEMA.keys():
        if col not in df_harmonized.columns:
            df_harmonized[col] = np.nan
    
    # Reorder columns to match schema
    df_harmonized = df_harmonized[[col for col in SHESENSE_SCHEMA.keys() if col in df_harmonized.columns]]
    
    return df_harmonized

def filter_female_subjects(df: pd.DataFrame, female_subjects: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter dataframe to female subjects only.
    
    Args:
        df: Input dataframe with 'subject' column
        female_subjects: List of female subject IDs. If None, assumes subjects starting with 'f' are female.
    
    Returns:
        Filtered dataframe
    """
    if female_subjects is None:
        # Default: assume subject IDs starting with 'f' are female (for Hongn dataset)
        female_mask = df['subject'].astype(str).str.startswith('f')
    else:
        female_mask = df['subject'].isin(female_subjects)
    
    filtered_df = df[female_mask].copy()
    logging.info(f"Filtered to {len(filtered_df)} rows from {len(df)} ({len(filtered_df['subject'].unique())} female subjects)")
    
    return filtered_df

def align_to_1hz(df: pd.DataFrame, 
                 timestamp_col: str = 'timestamp',
                 method: str = 'linear',
                 max_gap: str = '5s') -> pd.DataFrame:
    """
    Resample and align timeseries data to 1 Hz.
    
    Args:
        df: Input dataframe with timestamp column
        timestamp_col: Name of timestamp column
        method: Interpolation method ('linear', 'forward', 'backward')
        max_gap: Maximum gap to interpolate across
    
    Returns:
        Resampled dataframe at 1 Hz
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")
    
    # Set timestamp as index
    df_resampled = df.set_index(timestamp_col).copy()
    
    # Create 1 Hz time grid
    start_time = df_resampled.index.min()
    end_time = df_resampled.index.max()
    time_grid = pd.date_range(start=start_time, end=end_time, freq='1S')
    
    # Reindex to 1 Hz grid
    df_resampled = df_resampled.reindex(time_grid)
    
    # Interpolate missing values
    numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
    
    if method == 'linear':
        df_resampled[numeric_cols] = df_resampled[numeric_cols].interpolate(method='linear', limit_direction='both')
    elif method == 'forward':
        df_resampled[numeric_cols] = df_resampled[numeric_cols].fillna(method='ffill')
    elif method == 'backward':
        df_resampled[numeric_cols] = df_resampled[numeric_cols].fillna(method='bfill')
    
    # Handle categorical columns (like labels)
    categorical_cols = df_resampled.select_dtypes(include=['object', 'category']).columns
    df_resampled[categorical_cols] = df_resampled[categorical_cols].fillna(method='ffill')
    
    # Limit interpolation across gaps
    if max_gap:
        max_gap_seconds = pd.Timedelta(max_gap).total_seconds()
        for col in numeric_cols:
            # Identify large gaps and set them back to NaN
            gap_mask = df_resampled[col].isna()
            gap_starts = gap_mask & ~gap_mask.shift(1, fill_value=False)
            gap_ends = gap_mask & ~gap_mask.shift(-1, fill_value=False)
            
            for start_idx in df_resampled[gap_starts].index:
                end_idx = df_resampled[gap_ends & (df_resampled.index >= start_idx)].index[0]
                gap_duration = (end_idx - start_idx).total_seconds()
                
                if gap_duration > max_gap_seconds:
                    df_resampled.loc[start_idx:end_idx, col] = np.nan
    
    # Reset index to get timestamp as column
    df_resampled = df_resampled.reset_index().rename(columns={'index': timestamp_col})
    
    logging.info(f"Resampled to 1 Hz: {len(df_resampled)} samples over {(end_time - start_time).total_seconds()} seconds")
    
    return df_resampled

def validate_schema(df: pd.DataFrame, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate dataframe against SheSense schema.
    
    Args:
        df: Dataframe to validate
        strict: If True, all columns must be present and non-null
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required columns
    required_cols = ['timestamp', 'subject']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        issues.append(f"Missing required columns: {missing_required}")
    
    # Check data types
    for col, expected_dtype in SHESENSE_SCHEMA.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if expected_dtype.startswith('datetime') and not pd.api.types.is_datetime64_any_dtype(df[col]):
                issues.append(f"Column '{col}' should be datetime, got {actual_dtype}")
            elif expected_dtype == 'float64' and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' should be numeric, got {actual_dtype}")
            elif expected_dtype == 'str' and not pd.api.types.is_object_dtype(df[col]):
                issues.append(f"Column '{col}' should be string/object, got {actual_dtype}")
    
    # Check for completely null columns
    if strict:
        null_cols = [col for col in df.columns if df[col].isna().all()]
        if null_cols:
            issues.append(f"Completely null columns: {null_cols}")
    
    # Check timestamp ordering
    if 'timestamp' in df.columns and len(df) > 1:
        if not df['timestamp'].is_monotonic_increasing:
            issues.append("Timestamps are not monotonically increasing")
    
    return len(issues) == 0, issues

def get_data_quality_report(df: pd.DataFrame) -> Dict:
    """
    Generate data quality report for a SheSense dataframe.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'total_rows': len(df),
        'total_subjects': df['subject'].nunique() if 'subject' in df.columns else 0,
        'time_range': None,
        'missing_data': {},
        'sampling_rate': None
    }
    
    if 'timestamp' in df.columns and len(df) > 1:
        timestamps = pd.to_datetime(df['timestamp'])
        report['time_range'] = {
            'start': timestamps.min(),
            'end': timestamps.max(),
            'duration_hours': (timestamps.max() - timestamps.min()).total_seconds() / 3600
        }
        
        # Estimate sampling rate
        time_diffs = timestamps.diff().dt.total_seconds().dropna()
        median_interval = time_diffs.median()
        report['sampling_rate'] = f"{1/median_interval:.1f} Hz" if median_interval > 0 else "Unknown"
    
    # Missing data analysis
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        report['missing_data'][col] = {
            'count': missing_count,
            'percentage': missing_pct
        }
    
    return report
