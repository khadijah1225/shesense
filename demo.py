"""
Demonstration script for SheSense pipeline.

Shows basic usage of the key components.
"""

import sys
import os
sys.path.append('scripts')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
from data_contracts import harmonize_columns, filter_female_subjects, get_data_quality_report
from features import (generate_synthetic_hr, generate_synthetic_eda, generate_synthetic_temp,
                     compute_window_features, extract_features_from_windows)

def demo_synthetic_data_generation():
    """Demonstrate synthetic data generation."""
    print("=== Synthetic Data Generation Demo ===")
    
    # Generate 5 minutes of synthetic data
    duration = 300  # 5 minutes
    
    hr_data = generate_synthetic_hr(duration, base_hr=70, stress_factor=1.5)
    eda_data = generate_synthetic_eda(duration, base_eda=5.0, stress_factor=1.5)
    temp_data = generate_synthetic_temp(duration, base_temp=32.0)
    
    print(f"Generated {duration} seconds of synthetic data:")
    print(f"  HR: mean={hr_data.mean():.1f}, std={hr_data.std():.1f}")
    print(f"  EDA: mean={eda_data.mean():.1f}, std={eda_data.std():.1f}")
    print(f"  Temp: mean={temp_data.mean():.1f}, std={temp_data.std():.3f}")
    print()

def demo_create_mock_dataset():
    """Create a mock dataset for testing."""
    print("=== Mock Dataset Creation Demo ===")
    
    # Create timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i) for i in range(600)]  # 10 minutes
    
    # Create mock data for 2 subjects
    data_rows = []
    
    for subject_id in ['f01', 'f02']:
        for i, ts in enumerate(timestamps):
            # Simulate stress vs baseline periods
            is_stress = (i // 120) % 2 == 1  # Alternate every 2 minutes
            stress_factor = 2.0 if is_stress else 1.0
            
            # Generate physiological signals with stress effect
            hr = 70 + stress_factor * 5 + np.random.normal(0, 3)
            eda = 5 + stress_factor * 1.5 + np.random.exponential(0.3)
            temp = 32 + np.random.normal(0, 0.1)
            acc_mag = 1.0 + stress_factor * 0.3 + np.random.normal(0, 0.2)
            
            data_rows.append({
                'timestamp': ts,
                'subject': subject_id,
                'hr': hr,
                'eda_scl': eda,
                'temp': temp,
                'acc_mag': acc_mag,
                'label': 'stress' if is_stress else 'baseline'
            })
    
    df = pd.DataFrame(data_rows)
    print(f"Created mock dataset with {len(df)} samples")
    print(f"Subjects: {df['subject'].unique()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print()
    
    return df

def demo_data_validation(df):
    """Demonstrate data validation."""
    print("=== Data Validation Demo ===")
    
    # Generate quality report
    quality_report = get_data_quality_report(df)
    
    print("Data Quality Report:")
    print(f"  Total rows: {quality_report['total_rows']}")
    print(f"  Total subjects: {quality_report['total_subjects']}")
    print(f"  Sampling rate: {quality_report['sampling_rate']}")
    
    if quality_report['time_range']:
        duration_hours = quality_report['time_range']['duration_hours']
        print(f"  Duration: {duration_hours:.2f} hours")
    
    print("  Missing data:")
    for col, missing_info in quality_report['missing_data'].items():
        if missing_info['count'] > 0:
            print(f"    {col}: {missing_info['count']} ({missing_info['percentage']:.1f}%)")
    print()

def demo_feature_extraction(df):
    """Demonstrate feature extraction."""
    print("=== Feature Extraction Demo ===")
    
    # Extract features from 60-second windows
    features_df = extract_features_from_windows(
        df, 
        window_size_sec=60,
        hop_size_sec=30,
        sampling_rate=1.0
    )
    
    print(f"Extracted features from {len(features_df)} windows")
    print(f"Feature columns: {len([col for col in features_df.columns if col.startswith(('hrv_', 'eda_', 'temp_', 'acc_'))])}")
    
    # Show sample features for stress vs baseline
    stress_features = features_df[features_df['label'] == 'stress']
    baseline_features = features_df[features_df['label'] == 'baseline']
    
    if len(stress_features) > 0 and len(baseline_features) > 0:
        print("\nSample feature comparison (stress vs baseline):")
        
        feature_cols = ['hrv_hr_mean', 'hrv_rmssd', 'eda_mean', 'temp_mean']
        for col in feature_cols:
            if col in features_df.columns:
                stress_mean = stress_features[col].mean()
                baseline_mean = baseline_features[col].mean()
                print(f"  {col}: stress={stress_mean:.2f}, baseline={baseline_mean:.2f}")
    
    print()
    return features_df

def demo_subject_filtering():
    """Demonstrate female subject filtering."""
    print("=== Subject Filtering Demo ===")
    
    # Create mixed dataset
    mixed_data = pd.DataFrame({
        'subject': ['f01', 'f02', 'm01', 'm02', 'f03'],
        'hr': [70, 72, 75, 68, 71],
        'label': ['baseline'] * 5
    })
    
    print("Original dataset subjects:", mixed_data['subject'].tolist())
    
    # Filter to female subjects
    female_data = filter_female_subjects(mixed_data)
    print("After filtering to females:", female_data['subject'].tolist())
    print()

def main():
    """Run all demonstrations."""
    print("SheSense Pipeline Demonstration")
    print("=" * 50)
    print()
    
    # 1. Generate synthetic data
    demo_synthetic_data_generation()
    
    # 2. Subject filtering
    demo_subject_filtering()
    
    # 3. Create mock dataset
    mock_df = demo_create_mock_dataset()
    
    # 4. Data validation
    demo_data_validation(mock_df)
    
    # 5. Feature extraction
    features_df = demo_feature_extraction(mock_df)
    
    print("=== Demo Complete ===")
    print("Key components successfully demonstrated:")
    print("  ✓ Synthetic data generation")
    print("  ✓ Data validation and quality reporting")
    print("  ✓ Subject filtering")
    print("  ✓ Window-based feature extraction")
    print("  ✓ HRV, EDA, temperature, and accelerometer features")
    print()
    print("Next steps:")
    print("  1. Run `python scripts/extractdata.py` with your real datasets")
    print("  2. Use `python scripts/build_dataset.py` to harmonize data")
    print("  3. Create windowed features with `python scripts/make_windows.py`")
    print("  4. Train models with `python scripts/train_baseline.py`")

if __name__ == "__main__":
    main()
