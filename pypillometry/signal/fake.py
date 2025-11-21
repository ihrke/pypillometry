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
                       freq=0.5, filter_order=4, seed=None):
    """
    Generate smooth baseline pupil size via filtered noise.
    
    Creates pupil size time series by:
    1. Generating white noise
    2. Low-pass filtering at specified frequency with edge handling
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
    freq : float, default 0.5
        Cutoff frequency for low-pass filter (Hz), controls fluctuation speed.
        Typical values: 0.1-0.5 Hz for spontaneous fluctuations, 0.5-1 Hz for 
        cognitive load changes, 1-3 Hz for fast responses.
    filter_order : int, default 4
        Order of Butterworth filter. Higher order = sharper cutoff.
        Roll-off is (filter_order * 6) dB/octave.
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
    
    # Low-pass filter with edge handling to reduce transient artifacts
    nyquist = fs / 2
    b, a = sp_signal.butter(filter_order, freq / nyquist, btype='low')
    
    # Use padding to reduce edge effects
    # Pad length based on filter characteristics (3 cycles of lowest frequency)
    padlen = min(int(3 * fs / freq), n_samples - 1)
    filtered = sp_signal.filtfilt(b, a, noise, padlen=padlen, padtype='even')
    
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
        'filter_order': filter_order,
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


def generate_foreshortening_data(duration=120, fs=1000, eye='left',
                                theta=np.radians(95), phi=0.0, r=600, d=700,
                                A0_mean=3.5, A0_amplitude=0.5, A0_freq=0.5,
                                fixation_duration_mean=500, 
                                fixation_duration_std=100,
                                screen_resolution=(1920, 1080),
                                physical_screen_size=(520, 290),
                                screen_eye_distance=70.0,
                                measurement_noise=0.03,
                                seed=None, **kwargs):
    """
    Generate complete synthetic dataset for testing foreshortening correction algorithm.
    
    Orchestrates:
    1. Generate true pupil size A0(t) with slow fluctuations (~0.5 Hz by default)
    2. Generate fixation-based gaze positions (x, y)
    3. Apply foreshortening: measured_pupil = A0 * cos(alpha(x,y))
    4. Package into FakeEyeData with all metadata and ground truth
    
    Uses algorithm-specific names (A0, cos_alpha) since this function is specifically
    designed for testing the foreshortening correction algorithm.
    
    Parameters
    ----------
    duration : float
        Duration in seconds
    fs : float
        Sampling rate in Hz
    eye : str
        Which eye ('left' or 'right')
    theta, phi : float
        Camera position angles (radians)
    r : float
        Eye-to-camera distance (mm)
    d : float
        Eye-to-screen distance (mm)
    A0_mean : float
        Mean true pupil size
    A0_amplitude : float
        Amplitude of pupil fluctuations
    A0_freq : float
        Frequency of pupil fluctuations (Hz)
    fixation_duration_mean : float
        Mean fixation duration (ms)
    fixation_duration_std : float
        Std of fixation duration (ms)
    screen_resolution : tuple
        Screen size in pixels (width, height)
    physical_screen_size : tuple
        Physical screen size in mm (width, height)
    screen_eye_distance : float
        Eye-to-screen distance in cm (for EyeData metadata)
    measurement_noise : float
        Std of measurement noise added to pupil
    seed : int, optional
        Random seed for reproducibility
    **kwargs : additional parameters for FakeEyeData
    
    Returns
    -------
    data : FakeEyeData
        Synthetic eye-tracking data with:
        - sim_fct: reference to this function
        - sim_fct_name: 'generate_foreshortening_data'
        - sim_params: all generation parameters
        - sim_data: EyeDataDict with '{eye}_A0' and '{eye}_cosalpha' ground truth
    
    Examples
    --------
    >>> from pypillometry.signal.fake import generate_foreshortening_data
    >>> data = generate_foreshortening_data(duration=60, eye='left', seed=42)
    >>> 
    >>> # Access ground truth
    >>> A0_true = data.sim_data['left_A0']
    >>> cos_alpha_true = data.sim_data['left_cosalpha']
    >>> 
    >>> # Print generation call
    >>> print(data.get_generation_call())
    >>> 
    >>> # Fit and validate
    >>> calib = data.fit_foreshortening(eye='left', r=600, d=700)
    >>> theta_error = np.degrees(calib.theta - data.sim_params['theta'])
    >>> print(f"Theta estimation error: {theta_error:.2f}°")
    >>> 
    >>> # Regenerate with different seed
    >>> data2 = data.regenerate(seed=43)
    """
    # 1. Generate true pupil size A0(t)
    t, A0, _ = fake_pupil_baseline(
        duration, fs, mean=A0_mean, amplitude=A0_amplitude, 
        freq=A0_freq, seed=seed
    )
    
    # 2. Generate gaze positions x(t), y(t)
    _, x, y, _, _ = fake_gaze_fixations(
        duration, fs, fixation_duration_mean, fixation_duration_std,
        screen_bounds=((0, screen_resolution[0]), (0, screen_resolution[1])),
        seed=seed+1 if seed is not None else None
    )
    
    # 3. Compute cos_alpha and apply foreshortening
    from ..eyedata.foreshortening_calibration import _compute_cos_alpha_vectorized
    
    # Convert pixels to mm (centered coordinates)
    x_mm = (x - screen_resolution[0]/2) * physical_screen_size[0] / screen_resolution[0]
    y_mm = (y - screen_resolution[1]/2) * physical_screen_size[1] / screen_resolution[1]
    
    # Compute foreshortening factor
    cos_alpha = _compute_cos_alpha_vectorized(x_mm, y_mm, theta, phi, r, d, eye_offset=0.0)
    
    # Apply foreshortening: measured = A0 * cos(alpha)
    pupil_measured = A0 * cos_alpha
    
    # Add measurement noise
    if measurement_noise > 0:
        pupil_measured, _ = add_measurement_noise(
            pupil_measured, noise_level=measurement_noise,
            seed=seed+2 if seed is not None else None
        )
    
    # 4. Package into FakeEyeData
    from ..eyedata.fake_eyedata import FakeEyeData
    from ..eyedata.eyedatadict import EyeDataDict
    
    eye_data_kwargs = {f'{eye}_x': x, f'{eye}_y': y, f'{eye}_pupil': pupil_measured}
    
    # Store parameters (excluding function reference)
    sim_params = {
        'duration': duration,
        'fs': fs,
        'eye': eye,
        'theta': theta,
        'phi': phi,
        'r': r,
        'd': d,
        'A0_mean': A0_mean,
        'A0_amplitude': A0_amplitude,
        'A0_freq': A0_freq,
        'fixation_duration_mean': fixation_duration_mean,
        'fixation_duration_std': fixation_duration_std,
        'screen_resolution': screen_resolution,
        'physical_screen_size': physical_screen_size,
        'screen_eye_distance': screen_eye_distance,
        'measurement_noise': measurement_noise,
        'seed': seed,
    }
    
    # Store ground truth in EyeDataDict (use eye-specific keys)
    sim_data = EyeDataDict()
    sim_data[f'{eye}_A0'] = A0  # True pupil size
    sim_data[f'{eye}_cosalpha'] = cos_alpha  # Foreshortening factor
    
    return FakeEyeData(
        time=t,
        **eye_data_kwargs,
        sampling_rate=fs,
        screen_resolution=screen_resolution,
        physical_screen_size=physical_screen_size,
        screen_eye_distance=screen_eye_distance,
        sim_fct=generate_foreshortening_data,
        sim_fct_name='generate_foreshortening_data',
        sim_params=sim_params,
        sim_data=sim_data,
        **kwargs
    )


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

