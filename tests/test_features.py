"""
Unit tests for SheSense feature extraction functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from features import (
    compute_hrv_features, 
    compute_eda_features, 
    compute_temperature_features,
    compute_accelerometer_features,
    compute_window_features,
    generate_synthetic_hr,
    generate_synthetic_eda,
    generate_synthetic_temp
)

class TestHRVFeatures:
    """Test HRV feature computation."""
    
    def test_hrv_features_normal_data(self):
        """Test HRV features with normal physiological data."""
        # Generate synthetic HR data
        hr_data = generate_synthetic_hr(duration_sec=120, base_hr=70)
        
        features = compute_hrv_features(hr_data, sampling_rate=1.0)
        
        # Check that all expected features are present
        expected_features = [
            'hrv_hr_mean', 'hrv_hr_std', 'hrv_hr_min', 'hrv_hr_max', 'hrv_hr_range',
            'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50', 'hrv_lf_power', 'hrv_hf_power', 'hrv_lf_hf_ratio'
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"
        
        # Check reasonable values
        assert 60 <= features['hrv_hr_mean'] <= 80, "HR mean outside normal range"
        assert features['hrv_hr_std'] > 0, "HR std should be positive"
        assert features['hrv_rmssd'] >= 0, "RMSSD should be non-negative"
        assert features['hrv_sdnn'] >= 0, "SDNN should be non-negative"
    
    def test_hrv_features_empty_data(self):
        """Test HRV features with empty/invalid data."""
        empty_series = pd.Series([])
        features = compute_hrv_features(empty_series)
        
        # All features should be NaN
        for key, value in features.items():
            assert pd.isna(value), f"Feature {key} should be NaN for empty data"
    
    def test_hrv_features_constant_hr(self):
        """Test HRV features with constant HR (no variability)."""
        constant_hr = pd.Series([70.0] * 60)  # 1 minute of constant HR
        features = compute_hrv_features(constant_hr)
        
        assert features['hrv_hr_mean'] == 70.0
        assert features['hrv_hr_std'] == 0.0
        assert features['hrv_rmssd'] == 0.0
        assert features['hrv_pnn50'] == 0.0

class TestEDAFeatures:
    """Test EDA feature computation."""
    
    def test_eda_features_normal_data(self):
        """Test EDA features with normal data."""
        eda_data = generate_synthetic_eda(duration_sec=120, base_eda=5.0)
        
        features = compute_eda_features(eda_data, sampling_rate=1.0)
        
        expected_features = [
            'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range',
            'eda_slope', 'eda_peak_count', 'eda_peak_rate', 'eda_auc'
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"
        
        # Check reasonable values
        assert features['eda_mean'] > 0, "EDA mean should be positive"
        assert features['eda_std'] >= 0, "EDA std should be non-negative"
        assert features['eda_peak_count'] >= 0, "Peak count should be non-negative"
        assert features['eda_peak_rate'] >= 0, "Peak rate should be non-negative"
    
    def test_eda_features_with_stress(self):
        """Test EDA features with stress condition."""
        baseline_eda = generate_synthetic_eda(duration_sec=60, stress_factor=1.0)
        stress_eda = generate_synthetic_eda(duration_sec=60, stress_factor=3.0)
        
        baseline_features = compute_eda_features(baseline_eda)
        stress_features = compute_eda_features(stress_eda)
        
        # Stress should generally increase EDA levels and peaks
        assert stress_features['eda_mean'] > baseline_features['eda_mean'], \
            "Stress should increase EDA mean"

class TestTemperatureFeatures:
    """Test temperature feature computation."""
    
    def test_temp_features_normal_data(self):
        """Test temperature features with normal data."""
        temp_data = generate_synthetic_temp(duration_sec=300, base_temp=32.0)
        
        features = compute_temperature_features(temp_data)
        
        expected_features = [
            'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range',
            'temp_slope', 'temp_delta'
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"
        
        # Check reasonable values for skin temperature
        assert 30 <= features['temp_mean'] <= 35, "Temperature mean outside normal range"
        assert features['temp_std'] >= 0, "Temperature std should be non-negative"
    
    def test_temp_features_rising_temp(self):
        """Test temperature features with rising temperature."""
        # Create linearly increasing temperature
        temp_rising = pd.Series(np.linspace(32.0, 33.0, 100))
        features = compute_temperature_features(temp_rising)
        
        assert features['temp_slope'] > 0, "Slope should be positive for rising temperature"
        assert features['temp_delta'] > 0, "Delta should be positive for rising temperature"

class TestAccelerometerFeatures:
    """Test accelerometer feature computation."""
    
    def test_acc_features_normal_data(self):
        """Test accelerometer features with normal data."""
        # Generate synthetic accelerometer data
        n_samples = 100
        acc_x = pd.Series(np.random.normal(0, 0.1, n_samples))
        acc_y = pd.Series(np.random.normal(0, 0.1, n_samples))
        acc_z = pd.Series(np.random.normal(1, 0.1, n_samples))  # Z-axis should include gravity
        
        features = compute_accelerometer_features(acc_x, acc_y, acc_z, sampling_rate=1.0)
        
        expected_features = [
            'acc_mag_mean', 'acc_mag_std', 'acc_mag_min', 'acc_mag_max',
            'acc_activity_count', 'acc_dominant_freq'
        ]
        
        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"
        
        # Check reasonable values
        assert features['acc_mag_mean'] > 0, "Accelerometer magnitude mean should be positive"
        assert features['acc_activity_count'] >= 0, "Activity count should be non-negative"

class TestWindowFeatures:
    """Test window-level feature extraction."""
    
    def test_window_features_complete_data(self):
        """Test window feature extraction with complete data."""
        # Create synthetic window data
        window_size = 60
        timestamps = pd.date_range('2023-01-01', periods=window_size, freq='1s')
        
        window_df = pd.DataFrame({
            'timestamp': timestamps,
            'subject': 'test_subject',
            'hr': generate_synthetic_hr(duration_sec=window_size),
            'eda_scl': generate_synthetic_eda(duration_sec=window_size),
            'temp': generate_synthetic_temp(duration_sec=window_size),
            'acc_mag': np.random.normal(1, 0.2, window_size),
            'label': 'stress'
        })
        
        features = compute_window_features(window_df, 'test_window_001', sampling_rate=1.0)
        
        # Check basic window info
        assert features['window_id'] == 'test_window_001'
        assert features['subject'] == 'test_subject'
        assert features['label'] == 'stress'
        assert features['window_length'] == window_size
        
        # Check that HRV features are present
        hrv_features = [k for k in features.keys() if k.startswith('hrv_')]
        assert len(hrv_features) > 0, "Should have HRV features"
        
        # Check that EDA features are present
        eda_features = [k for k in features.keys() if k.startswith('eda_')]
        assert len(eda_features) > 0, "Should have EDA features"
        
        # Check that temperature features are present
        temp_features = [k for k in features.keys() if k.startswith('temp_')]
        assert len(temp_features) > 0, "Should have temperature features"
    
    def test_window_features_missing_data(self):
        """Test window feature extraction with missing data."""
        window_size = 60
        timestamps = pd.date_range('2023-01-01', periods=window_size, freq='1s')
        
        # Create window with some missing HR data
        hr_data = generate_synthetic_hr(duration_sec=window_size)
        hr_data.iloc[10:20] = np.nan  # Introduce missing data
        
        window_df = pd.DataFrame({
            'timestamp': timestamps,
            'subject': 'test_subject',
            'hr': hr_data,
            'label': 'baseline'
        })
        
        features = compute_window_features(window_df, 'test_window_002', sampling_rate=1.0)
        
        # Should still compute features despite missing data
        assert 'hrv_hr_mean' in features
        assert not pd.isna(features['hrv_hr_mean']), "Should compute HR mean despite some missing data"

class TestSyntheticDataGeneration:
    """Test synthetic data generation functions."""
    
    def test_synthetic_hr_generation(self):
        """Test synthetic HR data generation."""
        hr_data = generate_synthetic_hr(duration_sec=120, base_hr=70, stress_factor=1.0)
        
        assert len(hr_data) == 120, "Should generate correct number of samples"
        assert 60 <= hr_data.mean() <= 80, "HR should be in reasonable range"
        assert hr_data.std() > 0, "HR should have some variability"
    
    def test_synthetic_eda_generation(self):
        """Test synthetic EDA data generation."""
        eda_data = generate_synthetic_eda(duration_sec=120, base_eda=5.0, stress_factor=1.0)
        
        assert len(eda_data) == 120, "Should generate correct number of samples"
        assert (eda_data >= 0).all(), "EDA should be non-negative"
        assert eda_data.mean() > 0, "EDA should have positive mean"
    
    def test_synthetic_temp_generation(self):
        """Test synthetic temperature data generation."""
        temp_data = generate_synthetic_temp(duration_sec=120, base_temp=32.0)
        
        assert len(temp_data) == 120, "Should generate correct number of samples"
        assert 30 <= temp_data.mean() <= 35, "Temperature should be in reasonable range"

# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
