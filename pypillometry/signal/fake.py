"""
Synthetic data generation functions for testing and validation.

This module provides stateless functions for generating individual components
of eye-tracking data. Functions use abstract names (true_pupil, correction_factor)
to maintain generality. High-level orchestrator functions combine these generators
and use algorithm-specific names (e.g., A0, cos_alpha for foreshortening).
"""

import numpy as np
from scipy import signal as sp_signal


def fake_pupil_baseline(duration, fs, mean=3.5, amplitude=0.5, 
                       freq=3.0, seed=None):
    """
    Generate smooth baseline pupil size via filtered noise.
    
    Creates pupil size time series by:
    1. Generating white noise
    2. Low-pass filtering at specified frequency
    3. Scaling to desired amplitude and mean
    
    Parameters
    ----------
    duration : float
        Duration in seconds
    fs : float
        Sampling rate in Hz
    mean : float
        Mean pupil size (mm or arbitrary units)
    amplitude : float
        Amplitude of slow fluctuations
    freq : float
        Cutoff frequency for low-pass filter (Hz), controls fluctuation speed
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    t : np.ndarray
        Time vector (ms)
    true_pupil : np.ndarray
        True pupil size time series
    params : dict
        Parameters used for generation
    
    Examples
    --------
    >>> t, pupil, params = fake_pupil_baseline(duration=10, fs=1000, seed=42)
    >>> print(f"Mean: {pupil.mean():.2f}, Std: {pupil.std():.2f}")
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fs)
    
    # Generate white noise
    noise = rng.randn(n_samples)
    
    # Low-pass filter
    nyquist = fs / 2
    b, a = sp_signal.butter(4, freq / nyquist, btype='low')
    filtered = sp_signal.filtfilt(b, a, noise)
    
    # Scale to desired amplitude and mean
    filtered = filtered / np.std(filtered) * amplitude
    true_pupil = filtered + mean
    
    # Time vector in ms
    t = np.arange(n_samples) / fs * 1000
    
    params = {
        'duration': duration,
        'fs': fs,
        'mean': mean,
        'amplitude': amplitude,
        'freq': freq,
        'seed': seed,
    }
    
    return t, true_pupil, params


def add_measurement_noise(signal, noise_level=0.05, seed=None):
    """
    Add Gaussian measurement noise to a signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    noise_level : float
        Standard deviation of Gaussian noise to add
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    noisy_signal : np.ndarray
        Signal with added noise
    params : dict
        Parameters used for generation
    
    Examples
    --------
    >>> t, pupil, _ = fake_pupil_baseline(duration=10, fs=1000, seed=42)
    >>> noisy_pupil, params = add_measurement_noise(pupil, noise_level=0.1, seed=42)
    """
    rng = np.random.RandomState(seed)
    noisy_signal = signal + rng.normal(0, noise_level, size=signal.shape)
    
    params = {
        'noise_level': noise_level,
        'seed': seed,
    }
    
    return noisy_signal, params


def fake_gaze_fixations(duration, fs, fixation_duration_mean=300, 
                       fixation_duration_std=100, 
                       screen_bounds=((0, 1920), (0, 1080)),
                       n_fixations=None, seed=None):
    """
    Generate step-function gaze positions (fixations with random locations and durations).
    
    Creates gaze time series by:
    1. Determining fixation times (random durations)
    2. Selecting random fixation locations within screen bounds
    3. Creating step functions for x and y coordinates
    
    Parameters
    ----------
    duration : float
        Duration in seconds
    fs : float
        Sampling rate in Hz
    fixation_duration_mean : float
        Mean fixation duration (ms)
    fixation_duration_std : float
        Standard deviation of fixation duration (ms)
    screen_bounds : tuple of tuples
        ((xmin, xmax), (ymin, ymax)) screen boundaries in pixels
    n_fixations : int, optional
        Number of fixations (if None, determined by duration and mean fixation time)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    t : np.ndarray
        Time vector (ms)
    x : np.ndarray
        X gaze positions (pixels)
    y : np.ndarray
        Y gaze positions (pixels)
    fixation_info : dict
        Fixation times, positions, durations
    params : dict
        Parameters used for generation
    
    Examples
    --------
    >>> t, x, y, info, params = fake_gaze_fixations(duration=10, fs=1000, seed=42)
    >>> print(f"Number of fixations: {len(info['fixation_times'])}")
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration * fs)
    duration_ms = duration * 1000
    
    # Determine number of fixations
    if n_fixations is None:
        n_fixations = int(duration_ms / fixation_duration_mean) + 1
    
    # Generate fixation durations (ms)
    durations = rng.normal(fixation_duration_mean, fixation_duration_std, size=n_fixations)
    durations = np.clip(durations, 50, None)  # Minimum 50ms fixation
    
    # Adjust to fit duration
    total_duration = durations.sum()
    if total_duration > duration_ms:
        durations = durations * (duration_ms / total_duration)
    
    # Generate fixation positions
    x_bounds, y_bounds = screen_bounds
    x_positions = rng.uniform(x_bounds[0], x_bounds[1], size=n_fixations)
    y_positions = rng.uniform(y_bounds[0], y_bounds[1], size=n_fixations)
    
    # Create step functions
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    t = np.arange(n_samples) / fs * 1000
    
    fixation_times = []
    current_time = 0
    
    for i in range(n_fixations):
        start_idx = int(current_time * fs / 1000)
        end_time = current_time + durations[i]
        end_idx = min(int(end_time * fs / 1000), n_samples)
        
        if start_idx >= n_samples:
            break
        
        x[start_idx:end_idx] = x_positions[i]
        y[start_idx:end_idx] = y_positions[i]
        
        fixation_times.append((current_time, end_time, x_positions[i], y_positions[i]))
        current_time = end_time
    
    fixation_info = {
        'fixation_times': fixation_times,
        'n_fixations': len(fixation_times),
    }
    
    params = {
        'duration': duration,
        'fs': fs,
        'fixation_duration_mean': fixation_duration_mean,
        'fixation_duration_std': fixation_duration_std,
        'screen_bounds': screen_bounds,
        'n_fixations': n_fixations,
        'seed': seed,
    }
    
    return t, x, y, fixation_info, params

