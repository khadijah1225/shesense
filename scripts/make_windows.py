"""
Windowing and dataset creation for SheSense.

Segments physiological data into windows, computes features, and creates train/val/test splits.
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Dict, List
import logging

from features import extract_features_from_windows
from data_contracts import harmonize_columns, filter_female_subjects, align_to_1hz, validate_schema

def create_stratified_splits(features_df: pd.DataFrame, 
                           test_size: float = 0.2,
                           val_size: float = 0.2,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create subject-aware stratified train/validation/test splits.
    
    Args:
        features_df: DataFrame with extracted features
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed
    
    Returns:
        (train_df, val_df, test_df)
    """
    # Get unique subjects and their stress distributions
    subject_stats = features_df.groupby('subject').agg({
        'label': lambda x: (x == 'stress').mean(),  # proportion of stress windows
        'window_id': 'count'  # number of windows
    }).reset_index()
    subject_stats.columns = ['subject', 'stress_ratio', 'n_windows']
    
    # Filter subjects with sufficient data
    min_windows = 10  # minimum windows per subject
    valid_subjects = subject_stats[subject_stats['n_windows'] >= min_windows]['subject'].tolist()
    
    if len(valid_subjects) < 3:
        logging.warning(f"Only {len(valid_subjects)} subjects with sufficient data. Using simple split.")
        # Fallback to simple subject split
        subjects = features_df['subject'].unique()
        np.random.seed(random_state)
        np.random.shuffle(subjects)
        
        n_test = max(1, int(len(subjects) * test_size))
        n_val = max(1, int((len(subjects) - n_test) * val_size))
        
        test_subjects = subjects[:n_test]
        val_subjects = subjects[n_test:n_test + n_val]
        train_subjects = subjects[n_test + n_val:]
    else:
        # Subject-aware split with stratification
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        remaining_subjects, test_subjects = next(gss_test.split(valid_subjects, groups=valid_subjects))
        
        remaining_subjects = [valid_subjects[i] for i in remaining_subjects]
        test_subjects = [valid_subjects[i] for i in test_subjects]
        
        if len(remaining_subjects) > 1:
            val_size_adjusted = val_size / (1 - test_size)  # adjust for smaller pool
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
            train_subjects, val_subjects = next(gss_val.split(remaining_subjects, groups=remaining_subjects))
            
            train_subjects = [remaining_subjects[i] for i in train_subjects]
            val_subjects = [remaining_subjects[i] for i in val_subjects]
        else:
            train_subjects = remaining_subjects
            val_subjects = []
    
    # Create splits
    train_df = features_df[features_df['subject'].isin(train_subjects)].copy()
    val_df = features_df[features_df['subject'].isin(val_subjects)].copy() if val_subjects else pd.DataFrame()
    test_df = features_df[features_df['subject'].isin(test_subjects)].copy()
    
    logging.info(f"Dataset splits:")
    logging.info(f"  Train: {len(train_subjects)} subjects, {len(train_df)} windows")
    logging.info(f"  Val:   {len(val_subjects)} subjects, {len(val_df)} windows")
    logging.info(f"  Test:  {len(test_subjects)} subjects, {len(test_df)} windows")
    
    return train_df, val_df, test_df

def process_dataset(input_file: str,
                   output_dir: str,
                   window_size_sec: int = 60,
                   hop_size_sec: int = 30,
                   sampling_rate: float = 1.0,
                   female_only: bool = True,
                   resample_1hz: bool = True) -> Dict[str, str]:
    """
    Process raw physiological data into windowed features.
    
    Args:
        input_file: Path to input CSV/parquet file
        output_dir: Directory to save processed files
        window_size_sec: Window size in seconds
        hop_size_sec: Hop size in seconds
        sampling_rate: Target sampling rate
        female_only: Filter to female subjects only
        resample_1hz: Whether to resample to 1 Hz
    
    Returns:
        Dictionary with paths to generated files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)
    
    logging.info(f"Loaded {len(df)} rows from {input_file}")
    
    # Validate and harmonize schema
    is_valid, issues = validate_schema(df)
    if not is_valid:
        logging.warning(f"Schema validation issues: {issues}")
    
    # Filter to female subjects if requested
    if female_only and 'subject' in df.columns:
        df = filter_female_subjects(df)
    
    # Resample to 1 Hz if requested
    if resample_1hz and 'timestamp' in df.columns:
        logging.info("Resampling to 1 Hz...")
        df_list = []
        for subject, subject_df in df.groupby('subject'):
            resampled = align_to_1hz(subject_df, method='linear')
            df_list.append(resampled)
        df = pd.concat(df_list, ignore_index=True)
    
    # Extract features from windows
    logging.info(f"Extracting features with {window_size_sec}s windows, {hop_size_sec}s hops...")
    features_df = extract_features_from_windows(
        df, 
        window_size_sec=window_size_sec,
        hop_size_sec=hop_size_sec,
        sampling_rate=sampling_rate
    )
    
    # Create train/val/test splits
    train_df, val_df, test_df = create_stratified_splits(features_df)
    
    # Save datasets
    output_files = {}
    
    train_path = os.path.join(output_dir, 'train.parquet')
    train_df.to_parquet(train_path, index=False)
    output_files['train'] = train_path
    
    if not val_df.empty:
        val_path = os.path.join(output_dir, 'val.parquet')
        val_df.to_parquet(val_path, index=False)
        output_files['val'] = val_path
    
    test_path = os.path.join(output_dir, 'test.parquet')
    test_df.to_parquet(test_path, index=False)
    output_files['test'] = test_path
    
    # Save feature metadata
    feature_cols = [col for col in features_df.columns if col not in ['window_id', 'subject', 'label', 'timestamp_start', 'timestamp_end']]
    metadata = {
        'feature_columns': feature_cols,
        'window_size_sec': window_size_sec,
        'hop_size_sec': hop_size_sec,
        'sampling_rate': sampling_rate,
        'total_windows': len(features_df),
        'subjects': features_df['subject'].unique().tolist(),
        'label_distribution': features_df['label'].value_counts().to_dict()
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    output_files['metadata'] = metadata_path
    
    logging.info(f"Saved processed datasets to {output_dir}")
    return output_files

def main():
    parser = argparse.ArgumentParser(description="Create windowed feature datasets for SheSense")
    parser.add_argument('--input', required=True, help='Input CSV or parquet file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--window-sec', type=int, default=60, help='Window size in seconds')
    parser.add_argument('--hop-sec', type=int, default=30, help='Hop size in seconds')
    parser.add_argument('--sampling-rate', type=float, default=1.0, help='Sampling rate in Hz')
    parser.add_argument('--female-only', action='store_true', help='Filter to female subjects only')
    parser.add_argument('--no-resample', action='store_true', help='Skip resampling to 1 Hz')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Process dataset
    try:
        output_files = process_dataset(
            input_file=args.input,
            output_dir=args.output,
            window_size_sec=args.window_sec,
            hop_size_sec=args.hop_sec,
            sampling_rate=args.sampling_rate,
            female_only=args.female_only,
            resample_1hz=not args.no_resample
        )
        
        print("Successfully created windowed datasets:")
        for split, path in output_files.items():
            print(f"  {split}: {path}")
            
    except Exception as e:
        logging.error(f"Failed to process dataset: {e}")
        raise

if __name__ == "__main__":
    main()
