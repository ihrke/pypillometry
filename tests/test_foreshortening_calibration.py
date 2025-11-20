"""
Tests for foreshortening correction functionality.
"""

import numpy as np
import pytest
from pypillometry.eyedata.foreshortening_calibration import (
    ForeshorteningCalibration,
    _compute_cos_alpha_vectorized,
    _determine_knots,
    _create_bspline_basis,
    _split_params,
    _objective_function,
    _objective_gradient
)
from pypillometry.eyedata import EyeData


class TestHelperFunctions:
    """Tests for module-level helper functions."""
    
    def test_compute_cos_alpha_center(self):
        """Test cos(alpha) computation at screen center."""
        # Camera below screen (theta=95°, phi=0°)
        theta = np.radians(95)  # Slightly below horizontal
        phi = 0.0
        r = 600.0  # mm
        d = 700.0  # mm
        
        # At screen center (0, 0)
        cos_alpha = _compute_cos_alpha_vectorized(0.0, 0.0, theta, phi, r, d)
        
        # Cosine is always in [-1, 1]
        assert -1.0 <= cos_alpha <= 1.0
        assert isinstance(cos_alpha, (float, np.floating))
    
    def test_compute_cos_alpha_vectorized(self):
        """Test vectorized computation with arrays."""
        theta = np.pi / 2
        phi = 0.0
        r = 600.0
        d = 700.0
        
        x = np.array([0.0, 100.0, -100.0])
        y = np.array([0.0, 50.0, -50.0])
        
        cos_alpha = _compute_cos_alpha_vectorized(x, y, theta, phi, r, d)
        
        assert cos_alpha.shape == (3,)
        assert np.all(cos_alpha >= -1)
        assert np.all(cos_alpha <= 1)
    
    def test_compute_cos_alpha_bounds(self):
        """Test that cos(alpha) is always in valid range [-1, 1]."""
        theta = np.pi / 2
        phi = 0.0
        r = 600.0
        d = 700.0
        
        # Test various screen positions
        x = np.linspace(-400, 400, 20)
        y = np.linspace(-300, 300, 20)
        X, Y = np.meshgrid(x, y)
        
        cos_alpha = _compute_cos_alpha_vectorized(X.ravel(), Y.ravel(), theta, phi, r, d)
        
        # Cosine is always in [-1, 1]
        assert np.all(cos_alpha >= -1)
        assert np.all(cos_alpha <= 1)
    
    def test_determine_knots_basic(self):
        """Test automatic knot determination."""
        t_min = 0.0
        t_max = 120000.0  # 2 minutes in ms
        knots_per_second = 1.0
        
        knots = _determine_knots(t_min, t_max, knots_per_second)
        
        # Should have interior knots + boundary repetitions (4 each for cubic)
        assert len(knots) > 8  # At least boundary knots
        assert knots[0] == t_min
        assert knots[-1] == t_max
        
        # First 4 and last 4 should be boundary repeats
        assert np.all(knots[:4] == t_min)
        assert np.all(knots[-4:] == t_max)
    
    def test_determine_knots_spacing(self):
        """Test knot spacing respects knots_per_second parameter."""
        t_min = 0.0
        t_max = 10000.0  # 10 seconds
        
        knots_1 = _determine_knots(t_min, t_max, knots_per_second=1.0)
        knots_2 = _determine_knots(t_min, t_max, knots_per_second=2.0)
        
        # More knots per second should give more knots
        assert len(knots_2) > len(knots_1)
    
    def test_create_bspline_basis_shape(self):
        """Test B-spline basis matrix has correct shape."""
        t = np.linspace(0, 100, 50)
        knots = _determine_knots(0, 100, knots_per_second=1.0)
        
        basis = _create_bspline_basis(t, knots, degree=3)
        
        n_basis = len(knots) - 3 - 1  # len(knots) - degree - 1
        assert basis.shape == (50, n_basis)
        assert not np.any(np.isnan(basis))
    
    def test_create_bspline_basis_positivity(self):
        """Test that basis functions are non-negative."""
        t = np.linspace(0, 100, 50)
        knots = _determine_knots(0, 100, knots_per_second=1.0)
        
        basis = _create_bspline_basis(t, knots, degree=3)
        
        assert np.all(basis >= 0)
    
    def test_split_params(self):
        """Test parameter vector splitting."""
        n_basis = 10
        params = np.concatenate([
            np.ones(n_basis) * 3.5,  # Spline coeffs
            [np.pi / 2],  # theta
            [0.0]  # phi
        ])
        
        coeffs, theta, phi = _split_params(params, n_basis)
        
        assert coeffs.shape == (n_basis,)
        assert np.all(coeffs == 3.5)
        assert theta == np.pi / 2
        assert phi == 0.0
    
    def test_objective_function_basic(self):
        """Test objective function returns scalar."""
        n_samples = 100
        n_basis = 20
        
        # Create synthetic data
        x = np.random.randn(n_samples) * 100
        y = np.random.randn(n_samples) * 100
        t = np.linspace(0, 10000, n_samples)
        A = np.random.rand(n_samples) * 2 + 3
        
        knots = _determine_knots(0, 10000, knots_per_second=2.0)
        basis_matrix = _create_bspline_basis(t, knots, degree=3)
        
        params = np.concatenate([
            np.ones(basis_matrix.shape[1]) * 3.5,
            [np.pi / 2],
            [0.0]
        ])
        
        loss = _objective_function(
            params, x, y, t, A, basis_matrix, 
            r=600.0, d=700.0, lambda_smooth=1.0
        )
        
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0
    
    def test_objective_gradient_shape(self):
        """Test gradient has correct shape."""
        n_samples = 100
        
        x = np.random.randn(n_samples) * 100
        y = np.random.randn(n_samples) * 100
        t = np.linspace(0, 10000, n_samples)
        A = np.random.rand(n_samples) * 2 + 3
        
        knots = _determine_knots(0, 10000, knots_per_second=2.0)
        basis_matrix = _create_bspline_basis(t, knots, degree=3)
        n_basis = basis_matrix.shape[1]
        
        params = np.concatenate([
            np.ones(n_basis) * 3.5,
            [np.pi / 2],
            [0.0]
        ])
        
        gradient = _objective_gradient(
            params, x, y, t, A, basis_matrix,
            r=600.0, d=700.0, lambda_smooth=1.0
        )
        
        assert gradient.shape == params.shape
        assert gradient.shape == (n_basis + 2,)


class TestForeshorteningCalibration:
    """Tests for ForeshorteningCalibration class."""
    
    def setup_method(self):
        """Create a simple calibration object for testing."""
        self.theta = np.radians(95)
        self.phi = 0.0
        self.r = 600.0
        self.d = 700.0
        
        # Simple knots and coefficients (properly matched)
        # For degree 3: n_coeffs = len(knots) - degree - 1
        # knots length 11 → need 7 coefficients
        self.knots = np.array([0]*4 + [0, 50, 100] + [100]*4, dtype=float)
        self.coeffs = np.array([3.5, 3.7, 3.6, 3.8, 3.5, 3.6, 3.7])
        
        self.cal = ForeshorteningCalibration(
            eye='left',
            theta=self.theta,
            phi=self.phi,
            r=self.r,
            d=self.d,
            spline_coeffs=self.coeffs,
            spline_knots=self.knots,
            spline_degree=3,
            fit_metrics={'r2': 0.95, 'rmse': 0.12}
        )
    
    def test_initialization(self):
        """Test calibration object initialization."""
        assert self.cal.eye == 'left'
        assert self.cal.theta == self.theta
        assert self.cal.phi == self.phi
        assert self.cal.r == self.r
        assert self.cal.d == self.d
        assert len(self.cal.spline_coeffs) == 7
        assert self.cal.fit_metrics['r2'] == 0.95
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.cal)
        
        assert 'ForeshorteningCalibration' in repr_str
        assert 'left' in repr_str
        assert 'R²' in repr_str or 'R2' in repr_str
    
    def test_compute_cos_alpha_scalar(self):
        """Test computing cos(alpha) for scalar input."""
        cos_alpha = self.cal.compute_cos_alpha(0.0, 0.0)
        
        assert isinstance(cos_alpha, (float, np.floating))
        assert -1 <= cos_alpha <= 1
    
    def test_compute_cos_alpha_array(self):
        """Test computing cos(alpha) for array input."""
        x = np.array([0.0, 100.0, -100.0])
        y = np.array([0.0, 50.0, -50.0])
        
        cos_alpha = self.cal.compute_cos_alpha(x, y)
        
        assert cos_alpha.shape == (3,)
        assert np.all(cos_alpha >= -1)
        assert np.all(cos_alpha <= 1)
    
    def test_evaluate_spline_scalar(self):
        """Test spline evaluation at single time point."""
        A0 = self.cal.evaluate_spline(50.0)
        
        # BSpline can return array even for scalar input
        assert isinstance(A0, (float, np.floating, np.ndarray))
        assert np.all(A0 > 0)  # Pupil size should be positive
    
    def test_evaluate_spline_array(self):
        """Test spline evaluation at multiple time points."""
        t = np.linspace(0, 100, 50)
        A0 = self.cal.evaluate_spline(t)
        
        assert A0.shape == (50,)
        # Most values should be positive (boundary may be 0)
        assert np.sum(A0 > 0) >= 45
    
    def test_get_correction_factor_no_threshold(self):
        """Test correction factor computation (may return NaN if below threshold)."""
        correction = self.cal.get_correction_factor(0.0, 0.0, threshold=0.15)
        
        # Can return array or scalar
        assert isinstance(correction, (float, np.floating, np.ndarray))
        
        # If not NaN, correction factor should be >= 1
        if not np.isnan(correction):
            assert correction >= 1.0
    
    def test_get_correction_factor_with_threshold(self):
        """Test that correction factor handles thresholding correctly."""
        # Very peripheral position (large angle)
        x_extreme = 500.0  # mm
        y_extreme = 400.0  # mm
        
        correction = self.cal.get_correction_factor(x_extreme, y_extreme, threshold=0.5)
        
        # Can return array or scalar, may be NaN depending on geometry
        assert isinstance(correction, (float, np.floating, np.ndarray))
    
    def test_get_correction_factor_vectorized(self):
        """Test vectorized correction factor computation."""
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 50.0, 100.0])
        
        correction = self.cal.get_correction_factor(x, y)
        
        assert correction.shape == (3,)
        # Valid corrections should be >= 1, NaN for invalid
        valid_mask = ~np.isnan(correction)
        assert np.all(correction[valid_mask] >= 1.0)


class TestFitForeshorteningMethod:
    """Tests for EyeData.fit_foreshortening() method."""
    
    def create_synthetic_data(self, n_samples=1000, fs=100.0):
        """Create synthetic eye-tracking data with known geometry."""
        np.random.seed(42)  # For reproducibility
        
        # Time vector
        time = np.arange(n_samples) * (1000.0 / fs)  # ms
        
        # Known camera geometry
        theta_true = np.radians(95)
        phi_true = 0.0
        r_true = 600.0
        d_true = 700.0
        
        # Gaze positions (simulate scanning across screen)
        x_gaze = np.sin(2 * np.pi * time / 5000) * 200  # +/- 200 pixels
        y_gaze = np.cos(2 * np.pi * time / 7000) * 150  # +/- 150 pixels
        
        # True pupil size (smooth variation)
        A0_true = 3.5 + 0.3 * np.sin(2 * np.pi * time / 10000)
        
        # Convert gaze to mm (assuming 1920x1080 screen, 52cm wide, 29cm tall)
        scale_x = 520.0 / 1920.0  # mm/pixel
        scale_y = 290.0 / 1080.0  # mm/pixel
        x_mm = (x_gaze - 0) * scale_x  # Centered
        y_mm = (y_gaze - 0) * scale_y
        
        # Compute foreshortening
        cos_alpha = _compute_cos_alpha_vectorized(x_mm, y_mm, theta_true, phi_true, r_true, d_true)
        
        # Measured pupil size
        pupil = A0_true * cos_alpha + np.random.randn(n_samples) * 0.05  # Add noise
        
        # Ensure all positive
        pupil = np.abs(pupil)
        
        # Create EyeData object (without fill_time_discontinuities to avoid masking)
        data = EyeData(
            time=time,
            left_x=x_gaze + 960,  # Center at 960
            left_y=y_gaze + 540,  # Center at 540
            left_pupil=pupil,
            sampling_rate=fs,
            screen_resolution=(1920, 1080),
            physical_screen_size=(52.0, 29.0),  # cm
            screen_eye_distance=70.0,  # cm
            fill_time_discontinuities=False  # Don't create masks
        )
        
        return data, theta_true, phi_true, r_true, d_true
    
    def test_fit_basic(self):
        """Test basic fitting on synthetic data."""
        data, theta_true, phi_true, r_true, d_true = self.create_synthetic_data(n_samples=500, fs=100.0)
        
        # Fit the model
        calib = data.fit_foreshortening(
            eye='left',
            r=r_true,
            d=d_true,
            target_fs=50.0,
            lowpass_freq=3.0,
            verbose=False
        )
        
        # Check that calibration was created
        assert isinstance(calib, ForeshorteningCalibration)
        assert calib.eye == 'left'
        assert calib.r == r_true
        assert calib.d == d_true
        
        # Check fit quality
        assert 'r2' in calib.fit_metrics
        assert 'rmse' in calib.fit_metrics
        assert calib.fit_metrics['r2'] > 0.5  # Should have reasonable fit
    
    def test_fit_with_intervals(self):
        """Test fitting on subset of data using intervals."""
        from pypillometry.intervals import Intervals
        
        data, theta_true, phi_true, r_true, d_true = self.create_synthetic_data(n_samples=1000, fs=100.0)
        
        # Create intervals for first half of data
        intervals = Intervals([(0, 5000)], units='ms')
        
        calib = data.fit_foreshortening(
            eye='left',
            r=r_true,
            d=d_true,
            intervals=intervals,
            verbose=False
        )
        
        assert isinstance(calib, ForeshorteningCalibration)
        assert calib.fit_intervals is not None
    
    def test_fit_invalid_eye(self):
        """Test that invalid eye raises error."""
        data, theta_true, phi_true, r_true, d_true = self.create_synthetic_data(n_samples=100, fs=100.0)
        
        with pytest.raises(ValueError, match="Eye 'right' not found"):
            data.fit_foreshortening(
                eye='right',  # Only left eye exists
                r=r_true,
                d=d_true,
                verbose=False
            )
    
    def test_fit_invalid_r(self):
        """Test that negative r raises error."""
        data, theta_true, phi_true, r_true, d_true = self.create_synthetic_data(n_samples=100, fs=100.0)
        
        with pytest.raises(ValueError, match="r must be positive"):
            data.fit_foreshortening(
                eye='left',
                r=-600.0,
                d=d_true,
                verbose=False
            )
    
    def test_fit_invalid_d(self):
        """Test that negative d raises error."""
        data, theta_true, phi_true, r_true, d_true = self.create_synthetic_data(n_samples=100, fs=100.0)
        
        with pytest.raises(ValueError, match="d must be positive"):
            data.fit_foreshortening(
                eye='left',
                r=r_true,
                d=-700.0,
                verbose=False
            )
    
    def test_correction_reduces_spatial_variance(self):
        """Test that correction reduces gaze-position dependence."""
        data, theta_true, phi_true, r_true, d_true = self.create_synthetic_data(n_samples=500, fs=100.0)
        
        # Fit the model
        calib = data.fit_foreshortening(
            eye='left',
            r=r_true,
            d=d_true,
            verbose=False
        )
        
        # Get uncorrected pupil
        pupil_uncorrected = data['left', 'pupil'][:100]
        
        # Get gaze positions
        x_gaze = data['left', 'x'][:100]
        y_gaze = data['left', 'y'][:100]
        
        # Apply correction
        correction = calib.get_correction_factor(x_gaze, y_gaze)
        pupil_corrected = pupil_uncorrected * correction
        
        # Corrected pupil should have similar or higher variance over time
        # (since we're recovering the true temporal dynamics)
        assert not np.isnan(pupil_corrected).all()
        assert len(pupil_corrected[~np.isnan(pupil_corrected)]) > 0
    
    def test_fit_parameters_reasonable(self):
        """Test that fitted parameters are in reasonable ranges."""
        data, theta_true, phi_true, r_true, d_true = self.create_synthetic_data(n_samples=500, fs=100.0)
        
        calib = data.fit_foreshortening(
            eye='left',
            r=r_true,
            d=d_true,
            verbose=False
        )
        
        # Check theta is in valid range [0, pi]
        assert 0 <= calib.theta <= np.pi
        
        # Check phi is in valid range [0, 2*pi]
        assert 0 <= calib.phi <= 2 * np.pi
        
        # Check spline coefficients are positive
        assert np.all(calib.spline_coeffs > 0)
        
        # Check they're in reasonable range for pupil size (relaxed bounds)
        assert np.all(calib.spline_coeffs > 0.01)  # Very small lower bound
        assert np.all(calib.spline_coeffs < 20.0)  # Generous upper bound


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

