"""
Unit tests for SheSense data contracts and utilities.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from data_contracts import (
    harmonize_columns,
    filter_female_subjects,
    align_to_1hz,
    validate_schema,
    get_data_quality_report,
    SHESENSE_SCHEMA
)

class TestHarmonizeColumns:
    """Test column harmonization functions."""
    
    def test_harmonize_hongn_columns(self):
        """Test harmonization of Hongn dataset columns."""
        # Create mock Hongn data
        hongn_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1s'),
            'subject_id': ['f01'] * 100,
            'BVP_HR': np.random.normal(70, 10, 100),
            'EDA': np.random.normal(5, 1, 100),
            'TEMP': np.random.normal(32, 1, 100),
            'activity_label': ['baseline'] * 50 + ['stress'] * 50
        })
        
        harmonized = harmonize_columns(hongn_data, 'hongn')
        
        # Check that columns were renamed correctly
        assert 'subject' in harmonized.columns
        assert 'hr' in harmonized.columns
        assert 'eda_scl' in harmonized.columns
        assert 'temp' in harmonized.columns
        assert 'label' in harmonized.columns
        
        # Check that original columns are gone
        assert 'subject_id' not in harmonized.columns
        assert 'BVP_HR' not in harmonized.columns
    
    def test_harmonize_swell_columns(self):
        """Test harmonization of SWELL dataset columns."""
        swell_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1s'),
            'participant_id': ['p01'] * 100,
            'HR': np.random.normal(70, 10, 100),
            'RMSSD': np.random.normal(30, 10, 100),
            'SDNN': np.random.normal(40, 15, 100),
            'EDA_tonic': np.random.normal(5, 1, 100),
            'stress_label': ['low'] * 50 + ['high'] * 50
        })
        
        harmonized = harmonize_columns(swell_data, 'swell')
        
        assert 'subject' in harmonized.columns
        assert 'hr' in harmonized.columns
        assert 'rmssd' in harmonized.columns
        assert 'sdnn' in harmonized.columns
        assert 'eda_scl' in harmonized.columns
        assert 'label' in harmonized.columns
    
    def test_harmonize_unknown_dataset(self):
        """Test harmonization with unknown dataset type."""
        data = pd.DataFrame({'col1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Unknown dataset type"):
            harmonize_columns(data, 'unknown')

class TestFilterFemaleSubjects:
    """Test female subject filtering."""
    
    def test_filter_female_subjects_default(self):
        """Test filtering with default logic (subjects starting with 'f')."""
        data = pd.DataFrame({
            'subject': ['f01', 'f02', 'm01', 'm02', 'f03'],
            'value': [1, 2, 3, 4, 5]
        })
        
        filtered = filter_female_subjects(data)
        
        assert len(filtered) == 3
        assert all(subj.startswith('f') for subj in filtered['subject'])
        assert list(filtered['subject']) == ['f01', 'f02', 'f03']
    
    def test_filter_female_subjects_explicit_list(self):
        """Test filtering with explicit subject list."""
        data = pd.DataFrame({
            'subject': ['s01', 's02', 's03', 's04'],
            'value': [1, 2, 3, 4]
        })
        
        female_subjects = ['s01', 's03']
        filtered = filter_female_subjects(data, female_subjects)
        
        assert len(filtered) == 2
        assert list(filtered['subject']) == ['s01', 's03']

class TestAlignTo1Hz:
    """Test 1 Hz alignment function."""
    
    def test_align_regular_data(self):
        """Test alignment with already regular 1 Hz data."""
        # Create 1 Hz data
        timestamps = pd.date_range('2023-01-01', periods=60, freq='1s')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'hr': np.random.normal(70, 5, 60),
            'temp': np.random.normal(32, 0.5, 60)
        })
        
        aligned = align_to_1hz(data)
        
        assert len(aligned) == 60
        assert 'timestamp' in aligned.columns
        assert not aligned['hr'].isna().any()
    
    def test_align_irregular_data(self):
        """Test alignment with irregular sampling."""
        # Create irregular data (missing some samples)
        base_times = pd.date_range('2023-01-01', periods=60, freq='1s')
        # Remove every 5th timestamp to create gaps
        irregular_times = base_times[::5]  # Much sparser
        
        data = pd.DataFrame({
            'timestamp': irregular_times,
            'hr': np.random.normal(70, 5, len(irregular_times)),
            'temp': np.random.normal(32, 0.5, len(irregular_times))
        })
        
        aligned = align_to_1hz(data)
        
        # Should have interpolated to fill gaps
        assert len(aligned) >= len(irregular_times)
        assert 'timestamp' in aligned.columns
    
    def test_align_with_large_gaps(self):
        """Test alignment with large gaps that shouldn't be interpolated."""
        # Create data with a large gap
        times1 = pd.date_range('2023-01-01 00:00:00', periods=30, freq='1s')
        times2 = pd.date_range('2023-01-01 00:02:00', periods=30, freq='1s')  # 2-minute gap
        
        data = pd.DataFrame({
            'timestamp': list(times1) + list(times2),
            'hr': np.random.normal(70, 5, 60)
        })
        
        aligned = align_to_1hz(data, max_gap='30s')
        
        # Should have NaN values in the large gap
        assert aligned['hr'].isna().sum() > 0

class TestValidateSchema:
    """Test schema validation."""
    
    def test_valid_schema(self):
        """Test validation with valid schema."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1s'),
            'subject': ['f01'] * 100,
            'hr': np.random.normal(70, 10, 100),
            'eda_scl': np.random.normal(5, 1, 100),
            'temp': np.random.normal(32, 1, 100),
            'label': ['baseline'] * 100
        })
        
        is_valid, issues = validate_schema(data)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        data = pd.DataFrame({
            'hr': np.random.normal(70, 10, 100),
            'temp': np.random.normal(32, 1, 100)
        })
        
        is_valid, issues = validate_schema(data)
        
        assert not is_valid
        assert any('Missing required columns' in issue for issue in issues)
    
    def test_wrong_data_types(self):
        """Test validation with wrong data types."""
        data = pd.DataFrame({
            'timestamp': ['2023-01-01'] * 100,  # String instead of datetime
            'subject': ['f01'] * 100,
            'hr': ['70'] * 100,  # String instead of numeric
            'label': ['baseline'] * 100
        })
        
        is_valid, issues = validate_schema(data)
        
        assert not is_valid
        # Should have issues about data types
        assert len(issues) > 0

class TestDataQualityReport:
    """Test data quality reporting."""
    
    def test_quality_report_basic(self):
        """Test basic quality report generation."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1s'),
            'subject': ['f01'] * 500 + ['f02'] * 500,
            'hr': np.random.normal(70, 10, 1000),
            'eda_scl': np.concatenate([
                np.random.normal(5, 1, 900),
                [np.nan] * 100  # 10% missing data
            ]),
            'temp': np.random.normal(32, 1, 1000),
            'label': ['baseline'] * 500 + ['stress'] * 500
        })
        
        report = get_data_quality_report(data)
        
        assert 'total_rows' in report
        assert 'total_subjects' in report
        assert 'time_range' in report
        assert 'missing_data' in report
        assert 'sampling_rate' in report
        
        assert report['total_rows'] == 1000
        assert report['total_subjects'] == 2
        
        # Check missing data calculation
        assert 'eda_scl' in report['missing_data']
        assert report['missing_data']['eda_scl']['count'] == 100
        assert report['missing_data']['eda_scl']['percentage'] == 10.0
    
    def test_quality_report_no_timestamp(self):
        """Test quality report with no timestamp column."""
        data = pd.DataFrame({
            'subject': ['f01'] * 100,
            'hr': np.random.normal(70, 10, 100)
        })
        
        report = get_data_quality_report(data)
        
        assert report['time_range'] is None
        assert report['sampling_rate'] is None

# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
