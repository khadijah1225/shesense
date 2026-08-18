"""
Feature extraction functions for physiological signals in SheSense.

Computes HRV, EDA, temperature, and accelerometer features on sliding windows.
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def compute_hrv_features(hr_series: pd.Series, sampling_rate: float = 1.0) -> Dict[str, float]:
    """
    Compute heart rate variability features from HR time series.
    
    Args:
        hr_series: Heart rate time series (BPM)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of HRV features
    """
    features = {}
    
    # Basic HR statistics
    hr_clean = hr_series.dropna()
    if len(hr_clean) < 2:
        return {f'hrv_{k}': np.nan for k in ['hr_mean', 'hr_std', 'hr_min', 'hr_max', 'rmssd', 'sdnn', 'pnn50', 'lf_power', 'hf_power', 'lf_hf_ratio']}
    
    features['hrv_hr_mean'] = hr_clean.mean()
    features['hrv_hr_std'] = hr_clean.std()
    features['hrv_hr_min'] = hr_clean.min()
    features['hrv_hr_max'] = hr_clean.max()
    features['hrv_hr_range'] = features['hrv_hr_max'] - features['hrv_hr_min']
    
    # Convert HR to RR intervals (approximate)
    # Note: This is an approximation. Ideally we'd have actual RR intervals
    rr_intervals = 60.0 / hr_clean  # seconds
    rr_diff = np.diff(rr_intervals)
    
    # Time-domain HRV features
    if len(rr_diff) > 0:
        features['hrv_rmssd'] = np.sqrt(np.mean(rr_diff ** 2)) * 1000  # ms
        features['hrv_sdnn'] = np.std(rr_intervals) * 1000  # ms
        features['hrv_pnn50'] = np.sum(np.abs(rr_diff) > 0.05) / len(rr_diff) * 100  # %
    else:
        features['hrv_rmssd'] = np.nan
        features['hrv_sdnn'] = np.nan
        features['hrv_pnn50'] = np.nan
    
    # Frequency-domain HRV features using Welch's method
    try:
        if len(rr_intervals) >= 10:
            # Resample RR intervals to regular grid for spectral analysis
            time_regular = np.linspace(0, len(rr_intervals)/sampling_rate, len(rr_intervals))
            rr_interp = np.interp(np.linspace(0, len(rr_intervals)/sampling_rate, int(len(rr_intervals)*4)), 
                                 time_regular, rr_intervals)
            
            freqs, psd = signal.welch(rr_interp, fs=4.0, nperseg=min(256, len(rr_interp)//2))
            
            # Define frequency bands
            lf_band = (0.04, 0.15)  # Low frequency
            hf_band = (0.15, 0.4)   # High frequency
            
            lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs <= lf_band[1])])
            hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs <= hf_band[1])])
            
            features['hrv_lf_power'] = lf_power
            features['hrv_hf_power'] = hf_power
            features['hrv_lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else np.nan
        else:
            features['hrv_lf_power'] = np.nan
            features['hrv_hf_power'] = np.nan
            features['hrv_lf_hf_ratio'] = np.nan
    except:
        features['hrv_lf_power'] = np.nan
        features['hrv_hf_power'] = np.nan
        features['hrv_lf_hf_ratio'] = np.nan
    
    return features

def compute_eda_features(eda_series: pd.Series, sampling_rate: float = 1.0) -> Dict[str, float]:
    """
    Compute electrodermal activity features.
    
    Args:
        eda_series: EDA time series (μS)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of EDA features
    """
    features = {}
    
    eda_clean = eda_series.dropna()
    if len(eda_clean) < 2:
        return {f'eda_{k}': np.nan for k in ['mean', 'std', 'min', 'max', 'range', 'slope', 'peak_count', 'peak_rate', 'auc']}
    
    # Basic statistics
    features['eda_mean'] = eda_clean.mean()
    features['eda_std'] = eda_clean.std()
    features['eda_min'] = eda_clean.min()
    features['eda_max'] = eda_clean.max()
    features['eda_range'] = features['eda_max'] - features['eda_min']
    
    # Trend (slope)
    try:
        x = np.arange(len(eda_clean))
        slope, _ = np.polyfit(x, eda_clean.values, 1)
        features['eda_slope'] = slope
    except:
        features['eda_slope'] = np.nan
    
    # Peak detection for phasic component
    try:
        # Simple peak detection (could be improved with more sophisticated methods)
        peaks, _ = signal.find_peaks(eda_clean.values, 
                                   height=eda_clean.mean() + 0.5*eda_clean.std(),
                                   distance=int(sampling_rate))  # At least 1 second apart
        
        features['eda_peak_count'] = len(peaks)
        features['eda_peak_rate'] = len(peaks) / (len(eda_clean) / sampling_rate) * 60  # peaks per minute
        
        # Area under curve
        features['eda_auc'] = np.trapz(eda_clean.values)
    except:
        features['eda_peak_count'] = 0
        features['eda_peak_rate'] = 0
        features['eda_auc'] = np.nan
    
    return features

def compute_temperature_features(temp_series: pd.Series) -> Dict[str, float]:
    """
    Compute temperature features.
    
    Args:
        temp_series: Temperature time series (°C)
    
    Returns:
        Dictionary of temperature features
    """
    features = {}
    
    temp_clean = temp_series.dropna()
    if len(temp_clean) < 2:
        return {f'temp_{k}': np.nan for k in ['mean', 'std', 'min', 'max', 'range', 'slope', 'delta']}
    
    # Basic statistics
    features['temp_mean'] = temp_clean.mean()
    features['temp_std'] = temp_clean.std()
    features['temp_min'] = temp_clean.min()
    features['temp_max'] = temp_clean.max()
    features['temp_range'] = features['temp_max'] - features['temp_min']
    
    # Temperature change (slope)
    try:
        x = np.arange(len(temp_clean))
        slope, _ = np.polyfit(x, temp_clean.values, 1)
        features['temp_slope'] = slope
    except:
        features['temp_slope'] = np.nan
    
    # Temperature delta (end - start)
    features['temp_delta'] = temp_clean.iloc[-1] - temp_clean.iloc[0]
    
    return features

def compute_accelerometer_features(acc_x: pd.Series, acc_y: pd.Series, acc_z: pd.Series, 
                                 sampling_rate: float = 1.0) -> Dict[str, float]:
    """
    Compute accelerometer features.
    
    Args:
        acc_x, acc_y, acc_z: Accelerometer time series for each axis
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of accelerometer features
    """
    features = {}
    
    # Calculate magnitude
    try:
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        acc_mag_clean = acc_mag.dropna()
        
        if len(acc_mag_clean) < 2:
            return {f'acc_{k}': np.nan for k in ['mag_mean', 'mag_std', 'mag_min', 'mag_max', 'activity_count', 'dominant_freq']}
        
        # Basic magnitude statistics
        features['acc_mag_mean'] = acc_mag_clean.mean()
        features['acc_mag_std'] = acc_mag_clean.std()
        features['acc_mag_min'] = acc_mag_clean.min()
        features['acc_mag_max'] = acc_mag_clean.max()
        
        # Activity count (samples above threshold)
        threshold = acc_mag_clean.mean() + acc_mag_clean.std()
        features['acc_activity_count'] = np.sum(acc_mag_clean > threshold)
        
        # Dominant frequency
        try:
            freqs, psd = signal.welch(acc_mag_clean.values, fs=sampling_rate, nperseg=min(256, len(acc_mag_clean)//2))
            dominant_freq_idx = np.argmax(psd)
            features['acc_dominant_freq'] = freqs[dominant_freq_idx]
        except:
            features['acc_dominant_freq'] = np.nan
            
    except:
        features = {f'acc_{k}': np.nan for k in ['mag_mean', 'mag_std', 'mag_min', 'mag_max', 'activity_count', 'dominant_freq']}
    
    return features

def compute_window_features(window_df: pd.DataFrame, 
                          window_id: str,
                          sampling_rate: float = 1.0) -> Dict[str, float]:
    """
    Compute all features for a single window.
    
    Args:
        window_df: DataFrame containing one window of data
        window_id: Identifier for this window
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary containing all computed features
    """
    features = {'window_id': window_id}
    
    # Add basic window info
    features['window_length'] = len(window_df)
    features['subject'] = window_df['subject'].iloc[0] if 'subject' in window_df.columns else 'unknown'
    features['label'] = window_df['label'].mode().iloc[0] if 'label' in window_df.columns and not window_df['label'].isna().all() else 'unknown'
    
    if 'timestamp' in window_df.columns:
        features['timestamp_start'] = window_df['timestamp'].iloc[0]
        features['timestamp_end'] = window_df['timestamp'].iloc[-1]
    
    # HR/HRV features
    if 'hr' in window_df.columns:
        hrv_features = compute_hrv_features(window_df['hr'], sampling_rate)
        features.update(hrv_features)
    
    # EDA features
    if 'eda_scl' in window_df.columns:
        eda_features = compute_eda_features(window_df['eda_scl'], sampling_rate)
        features.update(eda_features)
    
    # Temperature features
    if 'temp' in window_df.columns:
        temp_features = compute_temperature_features(window_df['temp'])
        features.update(temp_features)
    
    # Accelerometer features (if individual axes available or magnitude)
    if 'acc_mag' in window_df.columns:
        # If magnitude is already computed
        acc_features = {
            'acc_mag_mean': window_df['acc_mag'].mean(),
            'acc_mag_std': window_df['acc_mag'].std(),
            'acc_mag_min': window_df['acc_mag'].min(),
            'acc_mag_max': window_df['acc_mag'].max(),
            'acc_activity_count': np.sum(window_df['acc_mag'] > (window_df['acc_mag'].mean() + window_df['acc_mag'].std())),
            'acc_dominant_freq': np.nan  # Would need spectral analysis
        }
        features.update(acc_features)
    elif all(col in window_df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
        # If individual axes are available
        acc_features = compute_accelerometer_features(
            window_df['acc_x'], window_df['acc_y'], window_df['acc_z'], sampling_rate
        )
        features.update(acc_features)
    
    return features

def extract_features_from_windows(df: pd.DataFrame, 
                                window_size_sec: int = 60,
                                hop_size_sec: int = 30,
                                sampling_rate: float = 1.0,
                                min_valid_ratio: float = 0.8) -> pd.DataFrame:
    """
    Extract features from sliding windows of physiological data.
    
    Args:
        df: Input dataframe with physiological signals
        window_size_sec: Window size in seconds
        hop_size_sec: Hop size in seconds (for overlap)
        sampling_rate: Sampling rate in Hz
        min_valid_ratio: Minimum ratio of valid (non-NaN) samples required per window
    
    Returns:
        DataFrame with extracted features
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")
    
    # Ensure timestamp is datetime and sorted
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate window and hop sizes in samples
    window_size_samples = int(window_size_sec * sampling_rate)
    hop_size_samples = int(hop_size_sec * sampling_rate)
    
    all_features = []
    
    # Group by subject to process each separately
    for subject, subject_df in df.groupby('subject'):
        subject_df = subject_df.reset_index(drop=True)
        
        # Create sliding windows
        for start_idx in range(0, len(subject_df) - window_size_samples + 1, hop_size_samples):
            end_idx = start_idx + window_size_samples
            window_df = subject_df.iloc[start_idx:end_idx].copy()
            
            # Check if window has enough valid data
            numeric_cols = window_df.select_dtypes(include=[np.number]).columns
            valid_ratio = window_df[numeric_cols].notna().mean().mean()
            
            if valid_ratio < min_valid_ratio:
                continue
            
            # Generate window ID
            window_id = f"{subject}_{start_idx:06d}_{end_idx:06d}"
            
            # Extract features
            try:
                window_features = compute_window_features(window_df, window_id, sampling_rate)
                all_features.append(window_features)
            except Exception as e:
                print(f"Warning: Failed to extract features for window {window_id}: {e}")
                continue
    
    if not all_features:
        raise ValueError("No valid windows found for feature extraction")
    
    features_df = pd.DataFrame(all_features)
    
    print(f"Extracted features from {len(features_df)} windows across {len(df['subject'].unique())} subjects")
    
    return features_df

# Utility functions for synthetic data generation (for testing)
def generate_synthetic_hr(duration_sec: int = 300, sampling_rate: float = 1.0, 
                         base_hr: float = 70, stress_factor: float = 1.0) -> pd.Series:
    """Generate synthetic heart rate data for testing."""
    n_samples = int(duration_sec * sampling_rate)
    time = np.linspace(0, duration_sec, n_samples)
    
    # Base rhythm + respiratory sinus arrhythmia + noise + stress effect
    hr = (base_hr + 
          3 * np.sin(2 * np.pi * 0.25 * time) +  # 0.25 Hz respiratory rhythm
          2 * np.sin(2 * np.pi * 0.1 * time) +   # 0.1 Hz slow rhythm
          stress_factor * 10 +                    # stress increases HR
          np.random.normal(0, 2, n_samples))      # noise
    
    return pd.Series(hr)

def generate_synthetic_eda(duration_sec: int = 300, sampling_rate: float = 1.0,
                          base_eda: float = 5.0, stress_factor: float = 1.0) -> pd.Series:
    """Generate synthetic EDA data for testing."""
    n_samples = int(duration_sec * sampling_rate)
    time = np.linspace(0, duration_sec, n_samples)
    
    # Tonic level + phasic responses + stress effect
    eda = (base_eda + 
           stress_factor * 2 +                          # stress increases tonic level
           np.random.exponential(0.5, n_samples) * stress_factor * 0.5 +  # phasic responses
           np.random.normal(0, 0.1, n_samples))         # noise
    
    return pd.Series(np.maximum(eda, 0))  # EDA can't be negative

def generate_synthetic_temp(duration_sec: int = 300, sampling_rate: float = 1.0,
                           base_temp: float = 32.0) -> pd.Series:
    """Generate synthetic temperature data for testing."""
    n_samples = int(duration_sec * sampling_rate)
    time = np.linspace(0, duration_sec, n_samples)
    
    # Slow drift + noise
    temp = (base_temp + 
            0.1 * np.sin(2 * np.pi * 0.01 * time) +    # very slow drift
            np.random.normal(0, 0.05, n_samples))       # noise
    
    return pd.Series(temp)
