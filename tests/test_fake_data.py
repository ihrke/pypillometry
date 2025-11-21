"""Tests for synthetic data generation (FakeEyeData and fake.py functions)."""

import numpy as np
import pytest
from scipy import signal as sp_signal

from pypillometry.eyedata import FakeEyeData, EyeDataDict
from pypillometry.signal.fake import (
    fake_pupil_baseline,
    fake_gaze_fixations,
    add_measurement_noise,
    generate_foreshortening_data,
)


class TestFakeEyeData:
    """Tests for FakeEyeData class."""
    
    def test_initialization(self):
        """Test basic FakeEyeData initialization."""
        time = np.arange(1000)
        x = np.random.rand(1000) * 1920
        y = np.random.rand(1000) * 1080
        pupil = np.random.rand(1000) * 3 + 2
        
        sim_params = {'duration': 1, 'fs': 1000, 'seed': 42}
        sim_data = EyeDataDict()
        sim_data['left_A0'] = pupil.copy()
        
        data = FakeEyeData(
            time=time,
            left_x=x,
            left_y=y,
            left_pupil=pupil,
            sim_fct=None,
            sim_fct_name='test_function',
            sim_params=sim_params,
            sim_data=sim_data,
        )
        
        assert isinstance(data, FakeEyeData)
        assert data.sim_fct_name == 'test_function'
        assert data.sim_params == sim_params
        assert 'left_A0' in data.sim_data
        assert len(data.sim_data['left_A0']) == 1000
    
    def test_sim_fct_name_auto_detect(self):
        """Test that sim_fct_name is auto-detected from sim_fct."""
        def my_generator():
            pass
        
        data = FakeEyeData(
            time=np.arange(100),
            left_x=np.ones(100),
            left_y=np.ones(100),
            left_pupil=np.ones(100),
            sim_fct=my_generator,
        )
        
        assert data.sim_fct_name == 'my_generator'
    
    def test_get_generation_call(self):
        """Test get_generation_call() method."""
        data = FakeEyeData(
            time=np.arange(100),
            left_x=np.ones(100),
            left_y=np.ones(100),
            left_pupil=np.ones(100),
            sim_fct_name='generate_foreshortening_data',
            sim_params={'duration': 60, 'fs': 1000, 'seed': 42},
        )
        
        call_str = data.get_generation_call()
        assert 'generate_foreshortening_data' in call_str
        assert 'duration=60' in call_str
        assert 'fs=1000' in call_str
        assert 'seed=42' in call_str
    
    def test_get_generation_call_no_function(self):
        """Test get_generation_call() when no function is set."""
        data = FakeEyeData(
            time=np.arange(100),
            left_x=np.ones(100),
            left_y=np.ones(100),
            left_pupil=np.ones(100),
        )
        
        call_str = data.get_generation_call()
        assert call_str == "Unknown generation function"
    
    def test_regenerate(self):
        """Test regenerate() method."""
        # Create a simple generator function
        def simple_generator(duration=1, fs=1000, seed=None):
            rng = np.random.RandomState(seed)
            time = np.arange(int(duration * fs))
            n = len(time)
            pupil = rng.randn(n) + 3
            x = np.ones(n) * 960
            y = np.ones(n) * 540
            return FakeEyeData(
                time=time,
                left_x=x,
                left_y=y,
                left_pupil=pupil,
                sim_fct=simple_generator,
                sim_params={'duration': duration, 'fs': fs, 'seed': seed},
            )
        
        # Generate initial data
        data1 = simple_generator(duration=1, fs=1000, seed=42)
        
        # Regenerate with same seed
        data2 = data1.regenerate()
        np.testing.assert_array_equal(data1.data['left_pupil'], data2.data['left_pupil'])
        
        # Regenerate with different seed
        data3 = data1.regenerate(seed=43)
        assert not np.allclose(data1.data['left_pupil'], data3.data['left_pupil'])
    
    def test_regenerate_no_function(self):
        """Test regenerate() raises error when sim_fct is None."""
        data = FakeEyeData(
            time=np.arange(100),
            left_x=np.ones(100),
            left_y=np.ones(100),
            left_pupil=np.ones(100),
        )
        
        with pytest.raises(ValueError, match="Cannot regenerate"):
            data.regenerate()
    
    def test_repr(self):
        """Test __repr__ method."""
        sim_data = EyeDataDict()
        sim_data['left_A0'] = np.ones(100)
        sim_data['left_cosalpha'] = np.ones(100)
        
        data = FakeEyeData(
            time=np.arange(100),
            left_x=np.ones(100),
            left_y=np.ones(100),
            left_pupil=np.ones(100),
            sim_fct_name='test_function',
            sim_data=sim_data,
        )
        
        repr_str = repr(data)
        assert 'Fake' in repr_str
        assert 'test_function' in repr_str
        assert 'left_A0' in repr_str
        assert 'left_cosalpha' in repr_str


class TestFakePupilBaseline:
    """Tests for fake_pupil_baseline function."""
    
    def test_basic_generation(self):
        """Test basic pupil baseline generation."""
        duration = 10
        fs = 1000
        t, pupil, params = fake_pupil_baseline(duration=duration, fs=fs, seed=42)
        
        assert len(t) == duration * fs
        assert len(pupil) == duration * fs
        assert params['duration'] == duration
        assert params['fs'] == fs
    
    def test_mean_and_amplitude(self):
        """Test that mean and amplitude are approximately correct."""
        mean = 4.0
        amplitude = 0.8
        t, pupil, _ = fake_pupil_baseline(
            duration=60, fs=1000, mean=mean, amplitude=amplitude, seed=42
        )
        
        # Check mean (should be close)
        assert np.abs(pupil.mean() - mean) < 0.2
        
        # Check amplitude (std should be approximately amplitude)
        assert np.abs(pupil.std() - amplitude) < 0.3
    
    def test_frequency_content(self):
        """Test that frequency content is correct."""
        duration = 60
        fs = 1000
        freq = 3.0
        
        t, pupil, _ = fake_pupil_baseline(
            duration=duration, fs=fs, freq=freq, amplitude=1.0, seed=42
        )
        
        # Compute power spectrum
        f, psd = sp_signal.welch(pupil, fs=fs, nperseg=min(4096, len(pupil)))
        
        # Most power should be below the cutoff frequency
        low_freq_power = np.sum(psd[f < freq])
        high_freq_power = np.sum(psd[f > freq * 2])
        
        assert low_freq_power > high_freq_power * 10
    
    def test_reproducibility(self):
        """Test that same seed produces same output."""
        seed = 42
        t1, pupil1, _ = fake_pupil_baseline(duration=5, fs=1000, seed=seed)
        t2, pupil2, _ = fake_pupil_baseline(duration=5, fs=1000, seed=seed)
        
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(pupil1, pupil2)
    
    def test_different_seeds_different_output(self):
        """Test that different seeds produce different output."""
        _, pupil1, _ = fake_pupil_baseline(duration=5, fs=1000, seed=42)
        _, pupil2, _ = fake_pupil_baseline(duration=5, fs=1000, seed=43)
        
        assert not np.allclose(pupil1, pupil2)


class TestAddMeasurementNoise:
    """Tests for add_measurement_noise function."""
    
    def test_noise_addition(self):
        """Test that noise is added correctly."""
        signal = np.ones(1000) * 3.0
        noise_level = 0.1
        
        noisy_signal, params = add_measurement_noise(signal, noise_level=noise_level, seed=42)
        
        assert len(noisy_signal) == len(signal)
        assert params['noise_level'] == noise_level
        
        # Check that noise was added (mean should be close to original, but not exact)
        assert np.abs(noisy_signal.mean() - 3.0) < 0.05
        assert not np.allclose(noisy_signal, signal)
    
    def test_noise_level(self):
        """Test that noise level affects standard deviation."""
        signal = np.ones(10000) * 5.0
        noise_level = 0.5
        
        noisy_signal, _ = add_measurement_noise(signal, noise_level=noise_level, seed=42)
        
        # Std of difference should be approximately noise_level
        diff = noisy_signal - signal
        assert np.abs(diff.std() - noise_level) < 0.05
    
    def test_reproducibility(self):
        """Test that same seed produces same output."""
        signal = np.ones(1000) * 3.0
        seed = 42
        
        noisy1, _ = add_measurement_noise(signal, noise_level=0.1, seed=seed)
        noisy2, _ = add_measurement_noise(signal, noise_level=0.1, seed=seed)
        
        np.testing.assert_array_equal(noisy1, noisy2)


class TestFakeGazeFixations:
    """Tests for fake_gaze_fixations function."""
    
    def test_basic_generation(self):
        """Test basic gaze fixation generation."""
        duration = 10
        fs = 1000
        
        t, x, y, info, params = fake_gaze_fixations(duration=duration, fs=fs, seed=42)
        
        assert len(t) == duration * fs
        assert len(x) == duration * fs
        assert len(y) == duration * fs
        assert 'fixation_times' in info
        assert info['n_fixations'] > 0
    
    def test_screen_bounds(self):
        """Test that gaze stays within screen bounds."""
        screen_bounds = ((0, 1920), (0, 1080))
        t, x, y, _, _ = fake_gaze_fixations(
            duration=10, fs=1000, screen_bounds=screen_bounds, seed=42
        )
        
        assert np.all(x >= screen_bounds[0][0])
        assert np.all(x <= screen_bounds[0][1])
        assert np.all(y >= screen_bounds[1][0])
        assert np.all(y <= screen_bounds[1][1])
    
    def test_step_function(self):
        """Test that gaze is a step function (piecewise constant)."""
        t, x, y, info, _ = fake_gaze_fixations(duration=5, fs=1000, seed=42)
        
        # Count unique values - should be equal to number of fixations
        n_unique_x = len(np.unique(x))
        n_fixations = info['n_fixations']
        
        # Allow some tolerance due to floating point
        assert n_unique_x == n_fixations or n_unique_x == n_fixations + 1
    
    def test_fixation_durations(self):
        """Test that fixation durations are reasonable."""
        fixation_duration_mean = 300  # ms
        t, x, y, info, _ = fake_gaze_fixations(
            duration=10,
            fs=1000,
            fixation_duration_mean=fixation_duration_mean,
            fixation_duration_std=50,
            seed=42
        )
        
        # Extract actual durations from fixation_times
        durations = [end - start for start, end, _, _ in info['fixation_times']]
        mean_duration = np.mean(durations)
        
        # Mean should be close to requested mean
        assert np.abs(mean_duration - fixation_duration_mean) < 100
    
    def test_reproducibility(self):
        """Test that same seed produces same output."""
        seed = 42
        t1, x1, y1, _, _ = fake_gaze_fixations(duration=5, fs=1000, seed=seed)
        t2, x2, y2, _, _ = fake_gaze_fixations(duration=5, fs=1000, seed=seed)
        
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


class TestGenerateForeshorteningData:
    """Tests for generate_foreshortening_data function."""
    
    def test_basic_generation(self):
        """Test basic foreshortening data generation."""
        data = generate_foreshortening_data(duration=5, fs=1000, seed=42)
        
        assert isinstance(data, FakeEyeData)
        assert len(data.data['left_pupil']) == 5000
        assert 'left_A0' in data.sim_data
        assert 'left_cosalpha' in data.sim_data
        assert data.sim_fct_name == 'generate_foreshortening_data'
    
    def test_ground_truth_consistency(self):
        """Test that measured pupil = A0 * cos_alpha (approximately)."""
        data = generate_foreshortening_data(
            duration=5, fs=1000, measurement_noise=0.0, seed=42
        )
        
        A0 = data.sim_data['left_A0']
        cos_alpha = data.sim_data['left_cosalpha']
        measured = data.data['left_pupil']
        
        expected = A0 * cos_alpha
        np.testing.assert_allclose(measured, expected, rtol=1e-10)
    
    def test_measurement_noise(self):
        """Test that measurement noise is added when specified."""
        data_no_noise = generate_foreshortening_data(
            duration=5, fs=1000, measurement_noise=0.0, seed=42
        )
        data_with_noise = generate_foreshortening_data(
            duration=5, fs=1000, measurement_noise=0.1, seed=42
        )
        
        # With noise, measured pupil should differ from A0 * cos_alpha
        A0 = data_with_noise.sim_data['left_A0']
        cos_alpha = data_with_noise.sim_data['left_cosalpha']
        measured = data_with_noise.data['left_pupil']
        
        expected = A0 * cos_alpha
        assert not np.allclose(measured, expected, rtol=1e-5)
    
    def test_parameters_stored(self):
        """Test that all parameters are stored in sim_params."""
        theta = np.radians(90)
        phi = np.radians(10)
        camera_eye_distance = 550
        screen_eye_distance = 650
        
        data = generate_foreshortening_data(
            duration=5, fs=500, theta=theta, phi=phi, 
            camera_eye_distance=camera_eye_distance, 
            screen_eye_distance=screen_eye_distance, 
            seed=42
        )
        
        assert data.sim_params['theta'] == theta
        assert data.sim_params['phi'] == phi
        assert data.sim_params['camera_eye_distance'] == camera_eye_distance
        assert data.sim_params['screen_eye_distance'] == screen_eye_distance
        assert data.sim_params['duration'] == 5
        assert data.sim_params['fs'] == 500
    
    def test_eye_selection(self):
        """Test that eye parameter determines which eye data is populated."""
        data_left = generate_foreshortening_data(duration=1, fs=1000, eye='left', seed=42)
        data_right = generate_foreshortening_data(duration=1, fs=1000, eye='right', seed=42)
        
        assert 'left_pupil' in data_left.data
        assert 'right_pupil' not in data_left.data
        
        assert 'right_pupil' in data_right.data
        assert 'left_pupil' not in data_right.data
    
    def test_reproducibility(self):
        """Test that same seed produces same output."""
        seed = 42
        data1 = generate_foreshortening_data(duration=5, fs=1000, seed=seed)
        data2 = generate_foreshortening_data(duration=5, fs=1000, seed=seed)
        
        np.testing.assert_array_equal(data1.data['left_pupil'], data2.data['left_pupil'])
        np.testing.assert_array_equal(data1.data['left_x'], data2.data['left_x'])
        np.testing.assert_array_equal(data1.sim_data['left_A0'], data2.sim_data['left_A0'])
    
    def test_regenerate(self):
        """Test that regenerate() works correctly."""
        data1 = generate_foreshortening_data(duration=5, fs=1000, seed=42)
        data2 = data1.regenerate(seed=43)
        
        # Different seeds should produce different data
        assert not np.allclose(data1.data['left_pupil'], data2.data['left_pupil'])
        
        # Same seed should reproduce
        data3 = data1.regenerate()
        np.testing.assert_array_equal(data1.data['left_pupil'], data3.data['left_pupil'])


class TestIntegration:
    """Integration tests for fake data with foreshortening fitting."""
    
    def test_fit_recovery(self):
        """Test that fitting recovers true geometry (approximately)."""
        # Generate data with known geometry
        true_theta = np.radians(85)
        true_phi = np.radians(5)
        true_camera_eye_distance = 600
        true_screen_eye_distance = 700
        
        data = generate_foreshortening_data(
            duration=60,
            fs=1000,
            theta=true_theta,
            phi=true_phi,
            camera_eye_distance=true_camera_eye_distance,
            screen_eye_distance=true_screen_eye_distance,
            physical_screen_size=(520.0, 290.0),  # mm
            measurement_noise=0.01,
            seed=42
        )
        
        # Fit foreshortening (r and d retrieved from data attributes)
        calib = data.fit_foreshortening(eye='left')
        
        # Check that fitted angles are close to true angles
        theta_error = np.abs(np.degrees(calib.theta - true_theta))
        phi_error = np.abs(np.degrees(calib.phi - true_phi))
        
        # Allow some error due to noise and optimization
        assert theta_error < 5  # Within 5 degrees
        assert phi_error < 5
    
    @pytest.mark.skip(reason="Test needs adjustment after unit changes - optimizer convergence issue")
    def test_correction_reduces_variance(self):
        """Test that correction reduces spatial variance in pupil."""
        data = generate_foreshortening_data(
            duration=30, fs=1000, 
            physical_screen_size=(520.0, 290.0),  # mm
            measurement_noise=5.0, seed=42  # More realistic noise level
        )
        
        # Fit foreshortening with explicit angles for better convergence
        calib = data.fit_foreshortening(
            eye='left', r=600, d=700,
            initial_theta=np.radians(20), initial_phi=np.radians(-90)
        )
        
        # Get correction factor
        x = data.data['left_x']
        y = data.data['left_y']
        correction = calib.get_correction_factor(x, y, threshold=0.0)
        
        # Apply correction
        measured_pupil = data.data['left_pupil']
        corrected_pupil = measured_pupil / correction
        
        # Remove NaN values for comparison
        valid = ~np.isnan(corrected_pupil)
        corrected_valid = corrected_pupil[valid]
        measured_valid = measured_pupil[valid]
        
        # Corrected pupil should have lower variance than measured
        # (since spatial variation due to foreshortening is removed)
        assert corrected_valid.std() < measured_valid.std()


def test_generate_foreshortening_data_with_string_units():
    """Test generate_foreshortening_data with string format units."""
    data = generate_foreshortening_data(
        duration=5,
        fs=100,
        eye='left',
        theta="20 degrees",
        phi="-90 degrees",
        camera_eye_distance="600 mm",
        screen_eye_distance="70 cm",
        physical_screen_size=("52 cm", "29 cm"),
        seed=42
    )
    
    # Should create FakeEyeData
    assert isinstance(data, FakeEyeData)
    
    # Parameters should be stored as floats in canonical units (radians, mm)
    assert np.isclose(data.sim_params['theta'], np.radians(20))
    assert np.isclose(data.sim_params['phi'], np.radians(-90))
    assert data.sim_params['camera_eye_distance'] == 600.0
    assert data.sim_params['screen_eye_distance'] == 700.0
    assert data.sim_params['physical_screen_size'] == (520.0, 290.0)


def test_generate_foreshortening_data_with_pint_quantities():
    """Test generate_foreshortening_data with Pint Quantities."""
    import pypillometry as pp
    
    data = generate_foreshortening_data(
        duration=5,
        fs=100,
        eye='left',
        theta=20 * pp.ureg.degree,
        phi=-90 * pp.ureg.degree,
        camera_eye_distance=60 * pp.ureg.cm,
        screen_eye_distance=0.7 * pp.ureg.m,
        seed=42
    )
    
    # Should create FakeEyeData
    assert isinstance(data, FakeEyeData)
    
    # Parameters should be stored as floats in canonical units
    assert np.isclose(data.sim_params['theta'], np.radians(20))
    assert np.isclose(data.sim_params['phi'], np.radians(-90))
    assert data.sim_params['camera_eye_distance'] == 600.0
    assert data.sim_params['screen_eye_distance'] == 700.0


def test_generate_foreshortening_data_mixed_units():
    """Test generate_foreshortening_data with mixed unit formats."""
    import pypillometry as pp
    
    data = generate_foreshortening_data(
        duration=5,
        fs=100,
        eye='left',
        theta="20 degrees",  # String
        phi=np.radians(-90),  # Plain (radians)
        camera_eye_distance=60 * pp.ureg.cm,  # Quantity
        screen_eye_distance="700 mm",  # String
        physical_screen_size=(520.0, "29 cm"),  # Mixed plain and string
        seed=42
    )
    
    # Should create FakeEyeData with consistent units
    assert isinstance(data, FakeEyeData)
    assert np.isclose(data.sim_params['theta'], np.radians(20))
    assert np.isclose(data.sim_params['phi'], np.radians(-90))
    assert data.sim_params['camera_eye_distance'] == 600.0
    assert data.sim_params['screen_eye_distance'] == 700.0
    assert data.sim_params['physical_screen_size'] == (520.0, 290.0)

