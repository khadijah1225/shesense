"""
Dataset harmonization and merging for SheSense.

Combines Hongn and SWELL datasets into a unified format.
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

from data_contracts import harmonize_columns, filter_female_subjects, align_to_1hz, get_data_quality_report

def load_hongn_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess Hongn et al. (Empatica E4) dataset.
    
    Args:
        file_path: Path to Hongn dataset CSV
    
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Map Hongn-specific columns if needed
    # This assumes the extractdata.py already processed the raw data
    if 'hr' not in df.columns and 'BVP_HR' in df.columns:
        df = df.rename(columns={'BVP_HR': 'hr'})
    
    # Extract subject ID from filename or create one
    if 'subject' not in df.columns:
        # Try to extract from filename
        filename = os.path.basename(file_path)
        if 'f' in filename:
            # Extract female subject ID (e.g., f01, f02, etc.)
            import re
            match = re.search(r'f\d+', filename)
            if match:
                df['subject'] = match.group()
            else:
                df['subject'] = 'unknown'
        else:
            df['subject'] = 'unknown'
    
    # Ensure female subjects are properly identified
    df = df[df['subject'].astype(str).str.startswith('f')].copy()
    
    # Harmonize columns
    df_harmonized = harmonize_columns(df, 'hongn')
    
    # Add dataset source
    df_harmonized['dataset'] = 'hongn'
    
    logging.info(f"Loaded Hongn data: {len(df_harmonized)} rows, {len(df_harmonized['subject'].unique())} subjects")
    
    return df_harmonized

def load_swell_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess SWELL-KW dataset.
    
    Args:
        file_path: Path to SWELL dataset CSV
    
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(file_path)
    
    # SWELL data is already processed by extractdata.py
    # Just need to harmonize column names
    df_harmonized = harmonize_columns(df, 'swell')
    
    # Add dataset source
    df_harmonized['dataset'] = 'swell'
    
    # Filter to female subjects (assuming we have this information)
    # For SWELL, we might need a separate metadata file or naming convention
    # For now, we'll keep all subjects but could add filtering later
    
    logging.info(f"Loaded SWELL data: {len(df_harmonized)} rows, {len(df_harmonized['subject'].unique())} subjects")
    
    return df_harmonized

def harmonize_stress_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize stress labels across datasets.
    
    Args:
        df: Input DataFrame with dataset-specific labels
    
    Returns:
        DataFrame with harmonized labels
    """
    df = df.copy()
    
    # Map various stress labels to standard categories
    label_mapping = {
        # Common stress labels
        'stress': 'stress',
        'stressed': 'stress',
        'high_stress': 'stress',
        'mental_stress': 'stress',
        'cognitive_load': 'stress',
        
        # Baseline/neutral labels
        'baseline': 'baseline',
        'rest': 'baseline',
        'relaxed': 'baseline',
        'calm': 'baseline',
        'neutral': 'baseline',
        'low_stress': 'baseline',
        
        # Physical activity (separate category)
        'physical': 'physical',
        'exercise': 'physical',
        'movement': 'physical',
        
        # Unknown/other
        'unknown': 'other',
        'other': 'other'
    }
    
    # Apply mapping (case-insensitive)
    df['label'] = df['label'].astype(str).str.lower()
    df['label'] = df['label'].map(label_mapping).fillna('other')
    
    # Log label distribution
    label_dist = df['label'].value_counts()
    logging.info(f"Label distribution after harmonization: {label_dist.to_dict()}")
    
    return df

def synchronize_timestamps(df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Synchronize timestamps across datasets.
    
    Args:
        df_list: List of DataFrames to synchronize
    
    Returns:
        List of synchronized DataFrames
    """
    synchronized_dfs = []
    
    for i, df in enumerate(df_list):
        df_sync = df.copy()
        
        # Ensure timestamp is datetime
        df_sync['timestamp'] = pd.to_datetime(df_sync['timestamp'])
        
        # Create relative timestamps from start of each subject's data
        subject_timestamps = []
        for subject, subject_df in df_sync.groupby('subject'):
            subject_df = subject_df.sort_values('timestamp').copy()
            start_time = subject_df['timestamp'].iloc[0]
            subject_df['timestamp_relative'] = (subject_df['timestamp'] - start_time).dt.total_seconds()
            subject_timestamps.append(subject_df)
        
        df_sync = pd.concat(subject_timestamps, ignore_index=True)
        synchronized_dfs.append(df_sync)
        
        logging.info(f"Dataset {i}: synchronized timestamps for {len(df_sync['subject'].unique())} subjects")
    
    return synchronized_dfs

def merge_datasets(hongn_df: pd.DataFrame, 
                  swell_df: pd.DataFrame,
                  resample_1hz: bool = True) -> pd.DataFrame:
    """
    Merge Hongn and SWELL datasets.
    
    Args:
        hongn_df: Hongn dataset DataFrame
        swell_df: SWELL dataset DataFrame
        resample_1hz: Whether to resample to 1 Hz
    
    Returns:
        Merged DataFrame
    """
    # Harmonize stress labels
    hongn_df = harmonize_stress_labels(hongn_df)
    swell_df = harmonize_stress_labels(swell_df)
    
    # Synchronize timestamps
    hongn_df, swell_df = synchronize_timestamps([hongn_df, swell_df])
    
    # Resample to 1 Hz if requested
    if resample_1hz:
        logging.info("Resampling datasets to 1 Hz...")
        
        # Process each dataset separately
        hongn_resampled = []
        for subject, subject_df in hongn_df.groupby('subject'):
            try:
                resampled = align_to_1hz(subject_df, method='linear')
                hongn_resampled.append(resampled)
            except Exception as e:
                logging.warning(f"Failed to resample Hongn subject {subject}: {e}")
        
        if hongn_resampled:
            hongn_df = pd.concat(hongn_resampled, ignore_index=True)
        
        swell_resampled = []
        for subject, subject_df in swell_df.groupby('subject'):
            try:
                resampled = align_to_1hz(subject_df, method='linear')
                swell_resampled.append(resampled)
            except Exception as e:
                logging.warning(f"Failed to resample SWELL subject {subject}: {e}")
        
        if swell_resampled:
            swell_df = pd.concat(swell_resampled, ignore_index=True)
    
    # Combine datasets
    combined_df = pd.concat([hongn_df, swell_df], ignore_index=True)
    
    # Sort by dataset, subject, and timestamp
    combined_df = combined_df.sort_values(['dataset', 'subject', 'timestamp']).reset_index(drop=True)
    
    logging.info(f"Merged datasets: {len(combined_df)} total rows")
    logging.info(f"Subjects by dataset: {combined_df.groupby('dataset')['subject'].nunique().to_dict()}")
    
    return combined_df

def create_quality_report(df: pd.DataFrame, output_dir: str):
    """Create data quality report for the merged dataset."""
    
    # Overall quality report
    overall_report = get_data_quality_report(df)
    
    # Per-dataset quality reports
    dataset_reports = {}
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        dataset_reports[dataset] = get_data_quality_report(dataset_df)
    
    # Per-subject reports
    subject_reports = []
    for subject in df['subject'].unique():
        subject_df = df[df['subject'] == subject]
        subject_report = get_data_quality_report(subject_df)
        subject_report['subject'] = subject
        subject_report['dataset'] = subject_df['dataset'].iloc[0]
        subject_reports.append(subject_report)
    
    # Combine reports
    quality_report = {
        'overall': overall_report,
        'by_dataset': dataset_reports,
        'by_subject': subject_reports,
        'summary': {
            'total_subjects': len(df['subject'].unique()),
            'total_samples': len(df),
            'datasets': df['dataset'].value_counts().to_dict(),
            'label_distribution': df['label'].value_counts().to_dict(),
            'time_coverage_hours': overall_report.get('time_range', {}).get('duration_hours', 0)
        }
    }
    
    # Save report
    import json
    with open(os.path.join(output_dir, 'data_quality_report.json'), 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    # Create summary visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Dataset distribution
        dataset_counts = df['dataset'].value_counts()
        axes[0, 0].pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Samples by Dataset')
        
        # Label distribution
        label_counts = df['label'].value_counts()
        axes[0, 1].bar(label_counts.index, label_counts.values)
        axes[0, 1].set_title('Label Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Missing data heatmap
        missing_data = df.select_dtypes(include=[np.number]).isnull().sum()
        if len(missing_data) > 0:
            axes[1, 0].bar(range(len(missing_data)), missing_data.values)
            axes[1, 0].set_xticks(range(len(missing_data)))
            axes[1, 0].set_xticklabels(missing_data.index, rotation=45)
            axes[1, 0].set_title('Missing Data by Feature')
            axes[1, 0].set_ylabel('Missing Count')
        
        # Subjects per dataset
        subj_per_dataset = df.groupby('dataset')['subject'].nunique()
        axes[1, 1].bar(subj_per_dataset.index, subj_per_dataset.values)
        axes[1, 1].set_title('Subjects per Dataset')
        axes[1, 1].set_ylabel('Number of Subjects')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_quality_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        logging.warning("Matplotlib not available. Skipping visualizations.")
    
    return quality_report

def main():
    parser = argparse.ArgumentParser(description="Harmonize and merge SheSense datasets")
    parser.add_argument('--hongn', help='Path to Hongn dataset CSV')
    parser.add_argument('--swell', help='Path to SWELL dataset CSV')
    parser.add_argument('--output', required=True, help='Output file path (parquet)')
    parser.add_argument('--female-only', action='store_true', help='Filter to female subjects only')
    parser.add_argument('--no-resample', action='store_true', help='Skip resampling to 1 Hz')
    parser.add_argument('--quality-report', action='store_true', help='Generate data quality report')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load datasets
    datasets = []
    
    if args.hongn:
        logging.info(f"Loading Hongn dataset from {args.hongn}")
        hongn_df = load_hongn_data(args.hongn)
        if args.female_only:
            hongn_df = filter_female_subjects(hongn_df)
        datasets.append(hongn_df)
    
    if args.swell:
        logging.info(f"Loading SWELL dataset from {args.swell}")
        swell_df = load_swell_data(args.swell)
        if args.female_only:
            # Note: SWELL filtering might need different logic
            # swell_df = filter_female_subjects(swell_df, female_subjects=[...])
            pass
        datasets.append(swell_df)
    
    if not datasets:
        raise ValueError("At least one dataset (--hongn or --swell) must be provided")
    
    # Merge datasets
    if len(datasets) == 2:
        logging.info("Merging Hongn and SWELL datasets")
        combined_df = merge_datasets(datasets[0], datasets[1], resample_1hz=not args.no_resample)
    else:
        # Single dataset
        combined_df = datasets[0]
        if not args.no_resample:
            logging.info("Resampling single dataset to 1 Hz")
            resampled_parts = []
            for subject, subject_df in combined_df.groupby('subject'):
                try:
                    resampled = align_to_1hz(subject_df, method='linear')
                    resampled_parts.append(resampled)
                except Exception as e:
                    logging.warning(f"Failed to resample subject {subject}: {e}")
            
            if resampled_parts:
                combined_df = pd.concat(resampled_parts, ignore_index=True)
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate quality report if requested
    if args.quality_report:
        report_dir = output_dir if output_dir else '.'
        logging.info("Generating data quality report")
        quality_report = create_quality_report(combined_df, report_dir)
        logging.info(f"Quality report saved to {report_dir}")
    
    # Save combined dataset
    combined_df.to_parquet(args.output, index=False)
    logging.info(f"Saved merged dataset to {args.output}")
    
    # Print summary
    print(f"Successfully merged {len(datasets)} dataset(s)")
    print(f"Total samples: {len(combined_df)}")
    print(f"Total subjects: {len(combined_df['subject'].unique())}")
    print(f"Datasets: {combined_df['dataset'].value_counts().to_dict()}")
    print(f"Labels: {combined_df['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
