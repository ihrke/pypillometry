"""
Tests for ExperimentalSetup class.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from pypillometry.eyedata import ExperimentalSetup


class TestExperimentalSetupInitialization:
    """Tests for ExperimentalSetup initialization."""
    
    def test_minimal_initialization(self):
        """Test that ExperimentalSetup can be created with no arguments."""
        setup = ExperimentalSetup()
        assert setup.screen_resolution is None
        assert setup.physical_screen_size is None
    
    def test_screen_resolution_only(self):
        """Test initialization with only screen resolution."""
        setup = ExperimentalSetup(screen_resolution=(1920, 1080))
        assert setup.screen_resolution == (1920, 1080)
        assert setup.screen_width == 1920
        assert setup.screen_height == 1080
    
    def test_physical_screen_size_mm(self):
        """Test initialization with physical screen size in mm (floats)."""
        setup = ExperimentalSetup(physical_screen_size=(520.0, 290.0))
        assert setup.physical_screen_width == 520.0
        assert setup.physical_screen_height == 290.0
    
    def test_physical_screen_size_string_mm(self):
        """Test initialization with physical screen size as string with mm."""
        setup = ExperimentalSetup(physical_screen_size=("520 mm", "290 mm"))
        assert setup.physical_screen_width == 520.0
        assert setup.physical_screen_height == 290.0
    
    def test_physical_screen_size_string_cm(self):
        """Test initialization with physical screen size as string with cm."""
        setup = ExperimentalSetup(physical_screen_size=("52 cm", "29 cm"))
        assert setup.physical_screen_width == 520.0
        assert setup.physical_screen_height == 290.0
    
    def test_eye_to_screen_perpendicular(self):
        """Test initialization with eye-to-screen distance."""
        setup = ExperimentalSetup(eye_to_screen_perpendicular="70 cm")
        assert setup.d == 700.0
    
    def test_eye_offset(self):
        """Test initialization with eye offset."""
        setup = ExperimentalSetup(
            eye_to_screen_perpendicular="70 cm",
            eye_offset=("3 cm", "-2 cm")
        )
        assert setup.delta_x == 30.0
        assert setup.delta_y == -20.0
        assert setup.eye_offset == (30.0, -20.0)
    
    def test_eye_to_screen_center_computes_d(self):
        """Test that eye_to_screen_center with offset computes correct d."""
        # If distance to center = 700 mm and offset = (30, 20), then
        # d = sqrt(700^2 - 30^2 - 20^2)
        setup = ExperimentalSetup(
            eye_to_screen_center="700 mm",
            eye_offset=("30 mm", "20 mm")
        )
        expected_d = np.sqrt(700**2 - 30**2 - 20**2)
        assert np.isclose(setup.d, expected_d)
    
    def test_screen_tilt_angles(self):
        """Test initialization with screen tilt angles."""
        setup = ExperimentalSetup(
            screen_pitch="-5 deg",
            screen_yaw="3 deg"
        )
        assert np.isclose(setup.alpha_tilt, np.radians(-5))
        assert np.isclose(setup.beta_tilt, np.radians(3))
        # Aliases
        assert np.isclose(setup.screen_pitch, np.radians(-5))
        assert np.isclose(setup.screen_yaw, np.radians(3))
    
    def test_camera_spherical_eye_frame(self):
        """Test initialization with camera spherical coords in eye frame."""
        setup = ExperimentalSetup(
            eye_to_screen_perpendicular="700 mm",
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            camera_position_relative_to="eye"
        )
        assert np.isclose(setup.r, 600.0)
        assert np.isclose(setup.theta, np.radians(20))
        assert np.isclose(setup.phi, np.radians(-90))
    
    def test_camera_offset_screen_frame(self):
        """Test initialization with camera offset in screen frame."""
        setup = ExperimentalSetup(
            eye_to_screen_perpendicular="700 mm",
            camera_offset=("0 mm", "-300 mm", "0 mm"),  # Below screen
            camera_position_relative_to="screen"
        )
        cam_screen = setup.camera_offset_screen_frame
        assert cam_screen == (0.0, -300.0, 0.0)
    
    def test_ipd(self):
        """Test initialization with inter-pupillary distance."""
        setup = ExperimentalSetup(ipd="63 mm")
        assert setup.ipd == 63.0
    
    def test_complete_setup(self):
        """Test initialization with all parameters."""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("52 cm", "29 cm"),
            eye_to_screen_perpendicular="70 cm",
            eye_offset=("0 mm", "0 mm"),
            screen_pitch="-5 deg",
            screen_yaw="0 deg",
            camera_spherical=("20 deg", "-90 deg", "60 cm"),
            camera_position_relative_to="eye",
            ipd="63 mm"
        )
        assert setup.screen_resolution == (1920, 1080)
        assert setup.d == 700.0
        assert setup.ipd == 63.0


class TestExperimentalSetupScreenProperties:
    """Tests for screen-related properties."""
    
    def setup_method(self):
        """Create a test setup."""
        self.setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm")
        )
    
    def test_screen_width(self):
        """Test screen_width property."""
        assert self.setup.screen_width == 1920
    
    def test_screen_height(self):
        """Test screen_height property."""
        assert self.setup.screen_height == 1080
    
    def test_physical_screen_width(self):
        """Test physical_screen_width property."""
        assert self.setup.physical_screen_width == 520.0
    
    def test_physical_screen_height(self):
        """Test physical_screen_height property."""
        assert self.setup.physical_screen_height == 290.0
    
    def test_screen_xlim(self):
        """Test screen_xlim property."""
        assert self.setup.screen_xlim == (0, 1920)
    
    def test_screen_ylim(self):
        """Test screen_ylim property."""
        assert self.setup.screen_ylim == (0, 1080)
    
    def test_mm_per_pixel_x(self):
        """Test mm_per_pixel_x conversion factor."""
        expected = 520.0 / 1920
        assert np.isclose(self.setup.mm_per_pixel_x, expected)
    
    def test_mm_per_pixel_y(self):
        """Test mm_per_pixel_y conversion factor."""
        expected = 290.0 / 1080
        assert np.isclose(self.setup.mm_per_pixel_y, expected)


class TestExperimentalSetupEyePosition:
    """Tests for eye position properties."""
    
    def test_eye_to_screen_center_calculation(self):
        """Test eye_to_screen_center property."""
        setup = ExperimentalSetup(
            eye_to_screen_perpendicular="700 mm",
            eye_offset=("30 mm", "40 mm")
        )
        # d = 700, delta_x = 30, delta_y = 40
        # distance = sqrt(700^2 + 30^2 + 40^2)
        expected = np.sqrt(700**2 + 30**2 + 40**2)
        assert np.isclose(setup.eye_to_screen_center, expected)
    
    def test_default_eye_offset_is_zero(self):
        """Test that default eye offset is (0, 0)."""
        setup = ExperimentalSetup(eye_to_screen_perpendicular="700 mm")
        assert setup.delta_x == 0.0
        assert setup.delta_y == 0.0
    
    def test_d_not_set_raises_error(self):
        """Test that accessing d when not set raises ValueError."""
        setup = ExperimentalSetup()
        with pytest.raises(ValueError, match="Eye-to-screen distance not set"):
            _ = setup.d


class TestExperimentalSetupCameraPosition:
    """Tests for camera position properties and transformations."""
    
    def test_camera_not_set_raises_error(self):
        """Test that accessing camera properties when not set raises ValueError."""
        setup = ExperimentalSetup(eye_to_screen_perpendicular="700 mm")
        with pytest.raises(ValueError, match="Camera position not set"):
            _ = setup.r
    
    def test_camera_spherical_to_r_theta_phi(self):
        """Test conversion from spherical input to r, theta, phi properties."""
        theta_deg, phi_deg, r_mm = 20, -90, 600
        setup = ExperimentalSetup(
            eye_to_screen_perpendicular="700 mm",
            camera_spherical=(f"{theta_deg} deg", f"{phi_deg} deg", f"{r_mm} mm"),
            camera_position_relative_to="eye"
        )
        assert np.isclose(setup.r, r_mm)
        assert np.isclose(setup.theta, np.radians(theta_deg))
        assert np.isclose(setup.phi, np.radians(phi_deg))
    
    def test_has_camera_position_true(self):
        """Test has_camera_position returns True when set."""
        setup = ExperimentalSetup(
            eye_to_screen_perpendicular="700 mm",
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            camera_position_relative_to="eye"
        )
        assert setup.has_camera_position()
    
    def test_has_camera_position_false(self):
        """Test has_camera_position returns False when not set."""
        setup = ExperimentalSetup(eye_to_screen_perpendicular="700 mm")
        assert not setup.has_camera_position()


class TestExperimentalSetupCoordinateConversion:
    """Tests for coordinate conversion methods."""
    
    def setup_method(self):
        """Create a test setup with pixel-to-mm conversion."""
        self.setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_to_screen_perpendicular="700 mm"
        )
    
    def test_pixels_to_mm_center(self):
        """Test conversion at screen center."""
        # Center of screen is at (960, 540) pixels
        x_mm, y_mm = self.setup.pixels_to_mm(960, 540)
        # Should be (0, 0) in mm (relative to screen center)
        assert np.isclose(x_mm, 0, atol=1e-10)
        assert np.isclose(y_mm, 0, atol=1e-10)
    
    def test_pixels_to_mm_corner(self):
        """Test conversion at screen corner."""
        # Top-left corner (0, 0) maps to (-W/2, -H/2) in mm coordinate system
        # (screen center is origin, y increases downward in this convention)
        x_mm, y_mm = self.setup.pixels_to_mm(0, 0)
        assert np.isclose(x_mm, -260.0)  # -520/2
        assert np.isclose(y_mm, -145.0)  # -290/2
    
    def test_pixels_to_mm_array(self):
        """Test conversion with arrays."""
        x_px = np.array([0, 960, 1920])
        y_px = np.array([0, 540, 1080])
        x_mm, y_mm = self.setup.pixels_to_mm(x_px, y_px)
        assert x_mm.shape == (3,)
        assert y_mm.shape == (3,)


class TestExperimentalSetupUnitParsing:
    """Tests for unit parsing functionality."""
    
    def test_distance_mm(self):
        """Test parsing distance in mm."""
        setup = ExperimentalSetup(eye_to_screen_perpendicular="700 mm")
        assert setup.d == 700.0
    
    def test_distance_cm(self):
        """Test parsing distance in cm."""
        setup = ExperimentalSetup(eye_to_screen_perpendicular="70 cm")
        assert setup.d == 700.0
    
    def test_distance_m(self):
        """Test parsing distance in meters."""
        setup = ExperimentalSetup(eye_to_screen_perpendicular="0.7 m")
        assert setup.d == 700.0
    
    def test_angle_degrees(self):
        """Test parsing angle in degrees."""
        setup = ExperimentalSetup(screen_pitch="5 degrees")
        assert np.isclose(setup.alpha_tilt, np.radians(5))
    
    def test_angle_deg(self):
        """Test parsing angle with 'deg' abbreviation."""
        setup = ExperimentalSetup(screen_pitch="5 deg")
        assert np.isclose(setup.alpha_tilt, np.radians(5))
    
    def test_angle_radians(self):
        """Test parsing angle in radians (string)."""
        setup = ExperimentalSetup(screen_pitch="0.1 rad")
        assert np.isclose(setup.alpha_tilt, 0.1)
    
    def test_plain_float_angle_assumed_radians(self):
        """Test that plain float angles are assumed to be radians."""
        setup = ExperimentalSetup(screen_pitch=0.1)
        assert np.isclose(setup.alpha_tilt, 0.1)
    
    def test_pint_quantity_distance(self):
        """Test parsing pint.Quantity for distance."""
        import pypillometry as pp
        setup = ExperimentalSetup(eye_to_screen_perpendicular=70 * pp.ureg.cm)
        assert setup.d == 700.0
    
    def test_pint_quantity_angle(self):
        """Test parsing pint.Quantity for angle."""
        import pypillometry as pp
        setup = ExperimentalSetup(screen_pitch=5 * pp.ureg.degree)
        assert np.isclose(setup.alpha_tilt, np.radians(5))


class TestExperimentalSetupValidation:
    """Tests for validation methods."""
    
    def test_has_screen_info_true(self):
        """Test has_screen_info when screen info is set."""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm")
        )
        assert setup.has_screen_info()
    
    def test_has_screen_info_partial(self):
        """Test has_screen_info when only resolution is set."""
        setup = ExperimentalSetup(screen_resolution=(1920, 1080))
        assert not setup.has_screen_info()
    
    def test_has_screen_info_false(self):
        """Test has_screen_info when nothing is set."""
        setup = ExperimentalSetup()
        assert not setup.has_screen_info()
    
    def test_is_complete_true(self):
        """Test is_complete when all required params are set."""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_to_screen_perpendicular="700 mm",
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            camera_position_relative_to="eye"
        )
        assert setup.is_complete()
    
    def test_is_complete_false(self):
        """Test is_complete when some params are missing."""
        setup = ExperimentalSetup(screen_resolution=(1920, 1080))
        assert not setup.is_complete()
    
    def test_validate_invalid_camera_relative_to(self):
        """Test validate catches invalid camera_position_relative_to."""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_to_screen_perpendicular="700 mm"
        )
        setup._camera_relative_to = "invalid"  # Directly set invalid value
        with pytest.raises(ValueError, match="camera_position_relative_to"):
            setup.validate()


class TestExperimentalSetupSerialization:
    """Tests for serialization methods."""
    
    def test_to_dict(self):
        """Test to_dict method."""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_to_screen_perpendicular="700 mm",
            ipd="63 mm"
        )
        d = setup.to_dict()
        assert d['screen_resolution'] == (1920, 1080)
        assert d['physical_screen_size'] == (520.0, 290.0)
        assert d['d'] == 700.0
        assert d['ipd'] == 63.0
    
    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            'screen_resolution': (1920, 1080),
            'physical_screen_size': (520.0, 290.0),
            'd': 700.0,
            'eye_offset': (0.0, 0.0),
            'alpha_tilt': 0.0,
            'beta_tilt': 0.0,
            'camera_position_relative_to': 'eye',
            'camera_spherical': (np.radians(20), np.radians(-90), 600.0),
            'ipd': 63.0,
        }
        setup = ExperimentalSetup.from_dict(data)
        assert setup.screen_resolution == (1920, 1080)
        assert setup.d == 700.0
        assert np.isclose(setup.r, 600.0)
        assert setup.ipd == 63.0
    
    def test_round_trip(self):
        """Test that to_dict -> from_dict preserves values."""
        original = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_to_screen_perpendicular="700 mm",
            eye_offset=("30 mm", "20 mm"),
            screen_pitch="-5 deg",
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            camera_position_relative_to="eye",
            ipd="63 mm"
        )
        
        d = original.to_dict()
        restored = ExperimentalSetup.from_dict(d)
        
        assert restored.screen_resolution == original.screen_resolution
        assert restored.d == original.d
        assert restored.delta_x == original.delta_x
        assert restored.delta_y == original.delta_y
        assert np.isclose(restored.alpha_tilt, original.alpha_tilt)
        assert np.isclose(restored.r, original.r)
        assert restored.ipd == original.ipd


class TestExperimentalSetupSummary:
    """Tests for summary method."""
    
    def test_summary_basic(self):
        """Test summary method returns dict."""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm")
        )
        summary = setup.summary()
        assert isinstance(summary, dict)
        assert 'screen_resolution' in summary
        assert summary['screen_resolution'] == (1920, 1080)
    
    def test_summary_with_camera(self):
        """Test summary includes camera info when set."""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_to_screen_perpendicular="700 mm",
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            camera_position_relative_to="eye"
        )
        summary = setup.summary()
        assert 'camera_distance_mm' in summary
        assert summary['camera_distance_mm'] == 600.0
    
    def test_summary_without_d(self):
        """Test summary when d is not set."""
        setup = ExperimentalSetup(screen_resolution=(1920, 1080))
        summary = setup.summary()
        assert summary['eye_to_screen_distance_mm'] == "not set"


class TestExperimentalSetupRepr:
    """Tests for string representation."""
    
    def test_repr_minimal(self):
        """Test __repr__ with minimal setup."""
        setup = ExperimentalSetup()
        repr_str = repr(setup)
        assert "ExperimentalSetup" in repr_str
    
    def test_repr_with_screen(self):
        """Test __repr__ includes screen info when set."""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm")
        )
        repr_str = repr(setup)
        assert "1920" in repr_str
        assert "520" in repr_str
    
    def test_repr_complete(self):
        """Test __repr__ with complete setup."""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_to_screen_perpendicular="700 mm",
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            camera_position_relative_to="eye"
        )
        repr_str = repr(setup)
        assert "ExperimentalSetup" in repr_str


class TestExperimentalSetupPlot:
    """Smoke tests for plotting method."""
    
    def setup_method(self):
        """Create test setup and close any existing plots."""
        self.setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_to_screen_perpendicular="700 mm",
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            camera_position_relative_to="eye"
        )
        plt.close('all')
    
    def teardown_method(self):
        """Clean up plots."""
        plt.close('all')
    
    def test_plot_3d(self):
        """Test 3D plot."""
        fig, ax = self.setup.plot(projection='3d')
        assert fig is not None
        assert ax is not None
    
    def test_plot_2d(self):
        """Test 2D plot (returns fig and 3 axes)."""
        result = self.setup.plot(projection='2d')
        # 2D plot returns fig and list of axes
        fig, axes = result
        assert fig is not None
        assert len(axes) == 3  # Three orthogonal views
    
    def test_plot_3d_with_viewing_angle(self):
        """Test 3D plot with custom viewing angle."""
        fig, ax = self.setup.plot(
            projection='3d',
            viewing_angle=("30 deg", "-60 deg")
        )
        assert fig is not None
        assert ax is not None
    
    def test_plot_2d_without_gaze_samples(self):
        """Test 2D plot without gaze samples."""
        fig, axes = self.setup.plot(
            projection='2d',
            show_gaze_samples=False
        )
        assert fig is not None
    
    def test_plot_2d_custom_gaze_samples(self):
        """Test 2D plot with custom gaze samples."""
        fig, axes = self.setup.plot(
            projection='2d',
            n_gaze_samples=16
        )
        assert fig is not None


class TestExperimentalSetupErrorHandling:
    """Tests for error handling."""
    
    def test_screen_width_without_resolution_raises(self):
        """Test that accessing screen_width without resolution raises error."""
        setup = ExperimentalSetup()
        with pytest.raises(ValueError, match="Screen resolution not set"):
            _ = setup.screen_width
    
    def test_physical_screen_width_without_size_raises(self):
        """Test that accessing physical_screen_width without size raises error."""
        setup = ExperimentalSetup()
        with pytest.raises(ValueError, match="Physical screen size not set"):
            _ = setup.physical_screen_width
    
    def test_mm_per_pixel_requires_both(self):
        """Test that mm_per_pixel requires both resolution and physical size."""
        setup = ExperimentalSetup(screen_resolution=(1920, 1080))
        with pytest.raises(ValueError, match="Physical screen size not set"):
            _ = setup.mm_per_pixel_x
    
    def test_pixels_to_mm_requires_setup(self):
        """Test that pixels_to_mm requires complete screen setup."""
        setup = ExperimentalSetup()
        with pytest.raises(ValueError):
            setup.pixels_to_mm(100, 100)


class TestExperimentalSetupCopy:
    """Tests for copying behavior via serialization."""
    
    def test_copy_via_serialization(self):
        """Test that to_dict/from_dict creates independent object (copy pattern)."""
        original = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_to_screen_perpendicular="700 mm"
        )
        
        # Copy via serialization round-trip
        copied = ExperimentalSetup.from_dict(original.to_dict())
        
        # Values should be equal
        assert copied.screen_resolution == original.screen_resolution
        assert copied.d == original.d
        
        # But they should be different objects
        assert copied is not original


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

