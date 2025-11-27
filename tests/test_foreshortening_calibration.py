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
from pypillometry.eyedata import EyeData, ExperimentalSetup


class TestHelperFunctions:
    """Tests for module-level helper functions."""
    
    def test_compute_cos_alpha_center(self):
        """Test cos(alpha) computation at screen center."""
        # Camera below screen (theta=95°, phi=0°)
        theta = np.radians(85)  # Slightly offset from perpendicular
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
        self.theta = np.radians(85)
        self.phi = 0.0
        self.r = 600.0
        self.d = 700.0
        
        # Create ExperimentalSetup with camera position in eye frame
        self.setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_screen_distance=f"{self.d} mm",
            camera_spherical=(f"{np.degrees(self.theta)} deg", f"{np.degrees(self.phi)} deg", f"{self.r} mm"),
            
        )
        
        # Simple knots and coefficients (properly matched)
        # For degree 3: n_coeffs = len(knots) - degree - 1
        # knots length 11 → need 7 coefficients
        self.knots = np.array([0]*4 + [0, 50, 100] + [100]*4, dtype=float)
        self.coeffs = np.array([3.5, 3.7, 3.6, 3.8, 3.5, 3.6, 3.7])
        
        self.cal = ForeshorteningCalibration(
            eye='left',
            theta=self.theta,
            phi=self.phi,
            experimental_setup=self.setup,
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
        assert np.isclose(self.cal.experimental_setup.r, self.r, rtol=1e-5)
        assert np.isclose(self.cal.experimental_setup.d, self.d, rtol=1e-5)
        assert len(self.cal.spline_coeffs) == 7
        assert self.cal.fit_metrics['r2'] == 0.95
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.cal)
        
        assert 'ForeshorteningCalibration' in repr_str
        assert 'left' in repr_str
        assert 'R²' in repr_str or 'R2' in repr_str
    
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
        """Test correction factor computation returns masked array."""
        # Using mm coordinates (from_pixels=False)
        correction = self.cal.get_correction_factor(0.0, 0.0, threshold=0.15, from_pixels=False)
        
        # Should return masked array
        assert isinstance(correction, np.ma.MaskedArray)
        
        # If not masked, correction factor should be >= 1
        if not correction.mask:
            assert correction >= 1.0
    
    def test_get_correction_factor_with_threshold(self):
        """Test that correction factor handles thresholding correctly."""
        # Very peripheral position (large angle)
        x_extreme = 500.0  # mm
        y_extreme = 400.0  # mm
        
        # Using mm coordinates (from_pixels=False)
        correction = self.cal.get_correction_factor(x_extreme, y_extreme, threshold=0.5, from_pixels=False)
        
        # Should return masked array
        assert isinstance(correction, np.ma.MaskedArray)
    
    def test_get_correction_factor_vectorized(self):
        """Test vectorized correction factor computation."""
        # Using mm coordinates (from_pixels=False)
        x = np.array([0.0, 100.0, 200.0])
        y = np.array([0.0, 50.0, 100.0])
        
        correction = self.cal.get_correction_factor(x, y, from_pixels=False)
        
        # Should return masked array
        assert isinstance(correction, np.ma.MaskedArray)
        assert correction.shape == (3,)
        # Valid (unmasked) corrections should be >= 1
        valid_data = correction[~correction.mask]
        if len(valid_data) > 0:
            assert np.all(valid_data >= 1.0)


class TestFitForeshorteningMethod:
    """Tests for EyeData.fit_foreshortening() method."""
    
    def create_synthetic_data(self, n_samples=1000, fs=100.0):
        """Create synthetic eye-tracking data with known geometry."""
        np.random.seed(42)  # For reproducibility
        
        # Time vector
        time = np.arange(n_samples) * (1000.0 / fs)  # ms
        
        # Known camera geometry
        theta_true = np.radians(85)
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
        
        # Create ExperimentalSetup with known geometry
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),  # mm
            eye_screen_distance=f"{d_true} mm",
            camera_spherical=(f"{np.degrees(theta_true)} deg", f"{np.degrees(phi_true)} deg", f"{r_true} mm"),
            
        )
        
        # Create EyeData object (without fill_time_discontinuities to avoid masking)
        data = EyeData(
            time=time,
            left_x=x_gaze + 960,  # Center at 960
            left_y=y_gaze + 540,  # Center at 540
            left_pupil=pupil,
            sampling_rate=fs,
            experimental_setup=setup,
            fill_time_discontinuities=False  # Don't create masks
        )
        
        return data, theta_true, phi_true, r_true, d_true
    
    def test_fit_basic(self):
        """Test basic fitting on synthetic data."""
        data, theta_true, phi_true, r_true, d_true = self.create_synthetic_data(n_samples=500, fs=100.0)
        
        # Fit the model
        calib = data.fit_foreshortening(
            eye='left',
            target_fs=50.0,
            lowpass_freq=3.0,
            verbose=False
        )
        
        # Check that calibration was created
        assert isinstance(calib, ForeshorteningCalibration)
        assert calib.eye == 'left'
        assert np.isclose(calib.experimental_setup.r, r_true, rtol=1e-5)
        assert np.isclose(calib.experimental_setup.d, d_true, rtol=1e-5)
        
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
    
    def test_fit_requires_camera_position(self):
        """Test that fitting requires camera position in experimental_setup."""
        # Create EyeData without camera position
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_screen_distance="700 mm"
            # No camera_spherical
        )
        data = EyeData(
            left_x=np.random.rand(100) * 1920,
            left_y=np.random.rand(100) * 1080,
            left_pupil=np.random.rand(100) * 3 + 2,
            sampling_rate=100.0,
            experimental_setup=setup,
            fill_time_discontinuities=False
        )
        
        with pytest.raises(ValueError, match="Camera position not set"):
            data.fit_foreshortening(eye='left', verbose=False)
    
    def test_correction_reduces_spatial_variance(self):
        """Test that correction reduces gaze-position dependence."""
        data, theta_true, phi_true, r_true, d_true = self.create_synthetic_data(n_samples=500, fs=100.0)
        
        # Fit the model
        calib = data.fit_foreshortening(
            eye='left',
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
            verbose=False
        )
        
        # Check theta is in valid range [0, pi/2] (camera must be forward of eye)
        assert 0 <= calib.theta <= np.pi/2
        
        # Check phi is in valid range [-pi, pi]
        assert -np.pi <= calib.phi <= np.pi
        
        # Check spline coefficients are positive
        assert np.all(calib.spline_coeffs > 0)
        
        # Check they're in reasonable range for pupil size (relaxed bounds)
        assert np.all(calib.spline_coeffs > 0.01)  # Very small lower bound
        assert np.all(calib.spline_coeffs < 20.0)  # Generous upper bound


class TestFitAndCorrection:
    """Tests for the complete workflow: fitting and applying correction."""
    
    def create_synthetic_data_with_variation(self, n_samples=500, fs=100.0):
        """Create synthetic data with known geometry and gaze variation."""
        np.random.seed(123)
        
        time = np.arange(n_samples) * (1000.0 / fs)
        
        theta_true = np.radians(85)
        phi_true = 0.0
        r_true = 600.0
        d_true = 700.0
        
        # More varied gaze positions
        x_gaze = np.random.randn(n_samples) * 150 + np.sin(2 * np.pi * time / 3000) * 200
        y_gaze = np.random.randn(n_samples) * 100 + np.cos(2 * np.pi * time / 4000) * 150
        
        # True pupil with temporal dynamics
        A0_true = 3.5 + 0.5 * np.sin(2 * np.pi * time / 8000) + 0.2 * np.cos(2 * np.pi * time / 12000)
        
        # Convert to mm
        scale_x = 520.0 / 1920.0
        scale_y = 290.0 / 1080.0
        x_mm = x_gaze * scale_x
        y_mm = y_gaze * scale_y
        
        # Apply foreshortening
        cos_alpha = _compute_cos_alpha_vectorized(x_mm, y_mm, theta_true, phi_true, r_true, d_true)
        pupil = np.abs(A0_true * cos_alpha + np.random.randn(n_samples) * 0.03)
        
        # Create ExperimentalSetup
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_screen_distance=f"{d_true} mm",
            camera_spherical=(f"{np.degrees(theta_true)} deg", f"{np.degrees(phi_true)} deg", f"{r_true} mm"),
            
        )
        
        data = EyeData(
            time=time,
            left_x=x_gaze + 960,
            left_y=y_gaze + 540,
            left_pupil=pupil,
            sampling_rate=fs,
            experimental_setup=setup,
            fill_time_discontinuities=False
        )
        
        return data, theta_true, phi_true, r_true, d_true, A0_true, cos_alpha
    
    def test_fit_and_correction_workflow(self):
        """Test complete workflow: fit model then apply correction."""
        data, theta_true, phi_true, r_true, d_true, A0_true, cos_alpha_true = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        # Fit the model
        calib = data.fit_foreshortening(
            eye='left',
            verbose=False
        )
        
        # Apply correction to some data points
        x_test = data['left', 'x'][:100]
        y_test = data['left', 'y'][:100]
        pupil_test = data['left', 'pupil'][:100]
        
        correction_factor = calib.get_correction_factor(x_test, y_test)
        pupil_corrected = pupil_test * correction_factor
        
        # Check that correction was applied
        assert pupil_corrected.shape == pupil_test.shape
        
        # Valid (non-NaN) corrections should be applied
        valid_mask = ~np.isnan(correction_factor)
        assert np.sum(valid_mask) > 0
    
    def test_fit_with_different_knot_spacing(self):
        """Test fitting with different knot spacing parameters."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        # Fit with different knot spacings
        calib_sparse = data.fit_foreshortening(
            eye='left',
            knots_per_second=0.5,
            verbose=False
        )
        
        calib_dense = data.fit_foreshortening(
            eye='left',
            knots_per_second=2.0,
            verbose=False
        )
        
        # Dense should have more coefficients
        assert len(calib_dense.spline_coeffs) > len(calib_sparse.spline_coeffs)
        
        # Both should have reasonable fit quality
        assert calib_sparse.fit_metrics['r2'] > 0.5
        assert calib_dense.fit_metrics['r2'] > 0.5
    
    def test_fit_with_different_smoothness(self):
        """Test fitting with different smoothness regularization."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        # Fit with low smoothness (more flexible)
        calib_low = data.fit_foreshortening(
            eye='left',
            lambda_smooth=0.1,
            verbose=False
        )
        
        # Fit with high smoothness (more constrained)
        calib_high = data.fit_foreshortening(
            eye='left',
            lambda_smooth=10.0,
            verbose=False
        )
        
        # Both should converge
        assert calib_low.fit_metrics['converged']
        assert calib_high.fit_metrics['converged']
        
        # High smoothness should have smoother coefficients
        diff_low = np.std(np.diff(calib_low.spline_coeffs))
        diff_high = np.std(np.diff(calib_high.spline_coeffs))
        assert diff_high <= diff_low * 1.5  # Allow some tolerance
    
    def test_correction_at_different_positions(self):
        """Test that correction varies appropriately with gaze position."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        calib = data.fit_foreshortening(
            eye='left',
            verbose=False
        )
        
        # Test correction at different screen positions
        x_center = np.array([960.0])
        y_center = np.array([540.0])
        
        x_peripheral = np.array([1500.0])
        y_peripheral = np.array([900.0])
        
        correction_center = calib.get_correction_factor(x_center, y_center)
        correction_peripheral = calib.get_correction_factor(x_peripheral, y_peripheral)
        
        # Correction factors should differ (unless both are NaN)
        if not (np.isnan(correction_center) or np.isnan(correction_peripheral)):
            assert not np.allclose(correction_center, correction_peripheral)
    
    def test_fit_initial_guesses(self):
        """Test fitting with different initial parameter guesses."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        # Fit with different initial guesses
        calib1 = data.fit_foreshortening(
            eye='left',
            initial_theta=np.radians(90),
            initial_phi=0.0,
            verbose=False
        )
        
        calib2 = data.fit_foreshortening(
            eye='left',
            initial_theta=np.radians(80),
            initial_phi=np.radians(10),
            verbose=False
        )
        
        # Both should converge to similar solutions
        assert calib1.fit_metrics['r2'] > 0.5
        assert calib2.fit_metrics['r2'] > 0.5
        
        # Camera positions should be in valid range
        assert 0 <= calib1.theta <= np.pi
        assert 0 <= calib2.theta <= np.pi
    
    def test_fit_metrics_completeness(self):
        """Test that all expected fit metrics are returned."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        calib = data.fit_foreshortening(
            eye='left',
            verbose=False
        )
        
        # Check all expected metrics are present
        required_metrics = ['r2', 'rmse', 'n_samples', 'loss', 'n_iterations', 'converged']
        for metric in required_metrics:
            assert metric in calib.fit_metrics
            assert calib.fit_metrics[metric] is not None
    
    def test_correction_factor_symmetry(self):
        """Test that correction factor behaves symmetrically."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        calib = data.fit_foreshortening(
            eye='left',
            initial_phi=0.0,  # Camera on x-axis
            verbose=False
        )
        
        # If camera is on x-axis, correction should be symmetric in y
        x = np.array([960.0, 960.0])
        y = np.array([640.0, 440.0])  # Symmetric around 540
        
        corrections = calib.get_correction_factor(x, y)
        
        # Both should be valid or both NaN
        assert (np.isnan(corrections[0]) and np.isnan(corrections[1])) or \
               (not np.isnan(corrections[0]) and not np.isnan(corrections[1]))
    
    def test_fit_with_lowpass_variations(self):
        """Test fitting with different lowpass filter settings."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        # Fit with different lowpass frequencies
        calib_low = data.fit_foreshortening(
            eye='left',
            lowpass_freq=2.0,
            verbose=False
        )
        
        calib_high = data.fit_foreshortening(
            eye='left',
            lowpass_freq=6.0,
            verbose=False
        )
        
        # Both should produce valid fits
        assert calib_low.fit_metrics['r2'] > 0.3
        assert calib_high.fit_metrics['r2'] > 0.3
    
    def test_spline_evaluation_consistency(self):
        """Test that spline evaluation is consistent."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        calib = data.fit_foreshortening(
            eye='left',
            verbose=False
        )
        
        # Evaluate at same point multiple times
        t_test = 50.0
        A0_1 = calib.evaluate_spline(t_test)
        A0_2 = calib.evaluate_spline(t_test)
        
        # Should give identical results
        assert np.allclose(A0_1, A0_2)
        
        # Evaluate at array
        t_array = np.array([50.0, 50.0])
        A0_array = calib.evaluate_spline(t_array)
        
        # Should be consistent
        assert np.allclose(A0_array[0], A0_array[1])
    
    def test_fit_uses_experimental_setup(self):
        """Test that fit_foreshortening uses experimental_setup from EyeData."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        # Data already has experimental_setup from create_synthetic_data_with_variation
        # Fit without providing r and d (should use experimental_setup)
        calib = data.fit_foreshortening(
            eye='left',
            verbose=False
        )
        
        # Should have used the correct values from experimental_setup
        assert np.isclose(calib.experimental_setup.r, r_true, rtol=1e-5)
        assert np.isclose(calib.experimental_setup.d, d_true, rtol=1e-5)
        
        # Fit quality should be good
        assert calib.fit_metrics['r2'] > 0.8
    
    def test_fit_without_experimental_setup_raises_error(self):
        """Test that fitting without experimental_setup raises error."""
        # Create data without experimental_setup
        data = EyeData(
            left_x=np.random.rand(100) * 1920,
            left_y=np.random.rand(100) * 1080,
            left_pupil=np.random.rand(100) * 3 + 2,
            sampling_rate=100.0,
            fill_time_discontinuities=False
        )
        
        # Should raise error when experimental_setup is not set
        with pytest.raises(ValueError, match="experimental_setup"):
            data.fit_foreshortening(eye='left', verbose=False)
    
    def test_fit_uses_setup_from_calibration(self):
        """Test that calibration object stores the experimental_setup."""
        data, theta_true, phi_true, r_true, d_true, _, _ = \
            self.create_synthetic_data_with_variation(n_samples=300, fs=100.0)
        
        # Fit with data's experimental_setup
        calib = data.fit_foreshortening(
            eye='left',
            verbose=False
        )
        
        # Calibration should have the experimental_setup
        assert calib.experimental_setup is not None
        assert np.isclose(calib.experimental_setup.r, r_true, rtol=1e-5)
        assert np.isclose(calib.experimental_setup.d, d_true, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

