"""
Smoke tests for plotting functions.
These tests verify that plotting functions run without errors.
"""
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np


class TestPlottingSmokeGeneric(unittest.TestCase):
    """Smoke tests for GenericPlotter methods"""
    
    def setUp(self):
        """Create test data once for all tests"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')  # Clean up before each test
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_timeseries_runs(self):
        """Test that plot_timeseries completes without error"""
        fig = plt.figure()
        self.data.plot.plot_timeseries()
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_timeseries_with_range(self):
        """Test plot_timeseries with custom range"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(plot_range=(0, 10))
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_timeseries_with_units(self):
        """Test plot_timeseries with different units"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(units="sec")
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_intervals_runs(self):
        """Test that plot_intervals completes without error using Intervals object"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        figs = self.data.plot.plot_intervals(intervals)
        self.assertIsNotNone(figs)
    
    def test_plot_timeseries_segments_runs(self):
        """Test that plot_timeseries_segments completes without error"""
        figs = self.data.plot.plot_timeseries_segments(interv=0.5)
        self.assertIsNotNone(figs)


class TestPlottingSmokePupil(unittest.TestCase):
    """Smoke tests for PupilPlotter methods"""
    
    def setUp(self):
        """Create test data once for all tests"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_pupil_plot_runs(self):
        """Test that pupil_plot completes without error"""
        fig = plt.figure()
        self.data.plot.pupil_plot()
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_with_eyes(self):
        """Test pupil_plot with specific eyes"""
        fig = plt.figure()
        self.data.plot.pupil_plot(eyes=['left'])
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_with_range(self):
        """Test pupil_plot with custom range"""
        fig = plt.figure()
        self.data.plot.pupil_plot(plot_range=(0, 10))
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_no_events(self):
        """Test pupil_plot without events"""
        fig = plt.figure()
        self.data.plot.pupil_plot(plot_events=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_no_blinks(self):
        """Test pupil_plot without blink highlighting"""
        fig = plt.figure()
        self.data.plot.pupil_plot(highlight_blinks=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_segments_runs(self):
        """Test that pupil_plot_segments completes without error"""
        figs = self.data.plot.pupil_plot_segments(interv=0.5)
        self.assertIsNotNone(figs)
    
    def test_plot_blinks_runs(self):
        """Test that plot_blinks completes without error"""
        figs = self.data.plot.plot_blinks()
        self.assertIsNotNone(figs)


class TestPlottingSmokeSimpleData(unittest.TestCase):
    """Smoke tests with simple, predictable data"""
    
    def setUp(self):
        """Create simple test data"""
        self.data = pp.PupilData(
            sampling_rate=100,
            left_pupil=np.sin(np.linspace(0, 10*np.pi, 1000)),
            right_pupil=np.cos(np.linspace(0, 10*np.pi, 1000))
        )
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_simple_pupil_plot(self):
        """Test pupil_plot with simple data"""
        fig = plt.figure()
        self.data.plot.pupil_plot()
        self.assertIsNotNone(plt.gcf())
    
    def test_simple_plot_timeseries(self):
        """Test plot_timeseries with simple data"""
        fig = plt.figure()
        self.data.plot.plot_timeseries()
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_masked_single_eye_variable(self):
        """Test plot_masked with single eye and variable"""
        self.data.pupil_blinks_detect()
        fig = plt.figure()
        result = self.data.plot.plot_masked('left', 'pupil')
        self.assertIsNone(result)  # Should return None
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_masked_multiple_eyes_no_merge(self):
        """Test plot_masked with multiple eyes without merging"""
        self.data.pupil_blinks_detect()
        # Only check if there are actual intervals
        intervals = self.data.mask_as_intervals(['left', 'right'], 'pupil')
        if sum(len(iv) for iv in intervals.values()) > 0:
            fig = plt.figure()
            result = self.data.plot.plot_masked(['left', 'right'], 'pupil', merge=False)
            self.assertIsNone(result)
            self.assertIsNotNone(plt.gcf())
            # Check that a legend was created (for multiple groups)
            ax = plt.gca()
            legend = ax.get_legend()
            self.assertIsNotNone(legend)
        else:
            # Just check it runs without error even with no intervals
            fig = plt.figure()
            self.data.plot.plot_masked(['left', 'right'], 'pupil', merge=False)
            self.assertIsNotNone(plt.gcf())
    
    def test_plot_masked_multiple_eyes_with_merge(self):
        """Test plot_masked with multiple eyes with merging"""
        self.data.pupil_blinks_detect()
        fig = plt.figure()
        result = self.data.plot.plot_masked(['left', 'right'], 'pupil', merge=True)
        self.assertIsNone(result)
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_masked_with_units_ms(self):
        """Test plot_masked with milliseconds units"""
        self.data.pupil_blinks_detect()
        # Check if there are intervals before checking labels
        intervals = self.data.mask_as_intervals('left', 'pupil')
        fig = plt.figure()
        self.data.plot.plot_masked('left', 'pupil', units='ms')
        self.assertIsNotNone(plt.gcf())
        # Only check x-axis label if there were intervals to plot
        if len(intervals) > 0:
            ax = plt.gca()
            xlabel = ax.get_xlabel()
            self.assertIn('ms', xlabel)
    
    def test_plot_masked_with_units_sec(self):
        """Test plot_masked with seconds units"""
        self.data.pupil_blinks_detect()
        # Check if there are intervals before checking labels
        intervals = self.data.mask_as_intervals('left', 'pupil')
        fig = plt.figure()
        self.data.plot.plot_masked('left', 'pupil', units='sec')
        self.assertIsNotNone(plt.gcf())
        # Only check x-axis label if there were intervals to plot
        if len(intervals) > 0:
            ax = plt.gca()
            xlabel = ax.get_xlabel()
            self.assertIn('sec', xlabel)
    
    def test_plot_masked_with_units_none(self):
        """Test plot_masked with indices (units=None)"""
        self.data.pupil_blinks_detect()
        # Check if there are intervals before checking labels
        intervals = self.data.mask_as_intervals('left', 'pupil')
        fig = plt.figure()
        self.data.plot.plot_masked('left', 'pupil', units=None)
        self.assertIsNotNone(plt.gcf())
        # Only check x-axis label if there were intervals to plot
        if len(intervals) > 0:
            ax = plt.gca()
            xlabel = ax.get_xlabel()
            self.assertIn('indices', xlabel)
    
    def test_plot_masked_no_labels(self):
        """Test plot_masked with show_labels=False"""
        self.data.pupil_blinks_detect()
        fig = plt.figure()
        self.data.plot.plot_masked(['left', 'right'], 'pupil', show_labels=False)
        self.assertIsNotNone(plt.gcf())
        # Title and legend should not be present when show_labels=False
        ax = plt.gca()
        title = ax.get_title()
        self.assertEqual(title, '')
    
    def test_plot_masked_empty_mask(self):
        """Test plot_masked with no masked data"""
        # Don't detect blinks, so masks should be mostly empty
        fig = plt.figure()
        # This should handle empty intervals gracefully
        self.data.plot.plot_masked('left', 'pupil')
        # Should complete without error even with no intervals
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_masked_all_eyes_all_variables(self):
        """Test plot_masked with default parameters (all eyes and variables)"""
        self.data.pupil_blinks_detect()
        fig = plt.figure()
        self.data.plot.plot_masked([], [], merge=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_masked_with_kwargs(self):
        """Test plot_masked accepts additional kwargs for matplotlib"""
        self.data.pupil_blinks_detect()
        fig = plt.figure()
        self.data.plot.plot_masked('left', 'pupil', linewidth=5, alpha=0.5)
        self.assertIsNotNone(plt.gcf())


class TestPlottingExperimentalSetup(unittest.TestCase):
    """Smoke tests for plot_experimental_setup"""
    
    def setUp(self):
        """Create test data with experimental setup parameters"""
        from pypillometry.eyedata import ExperimentalSetup
        
        # Create minimal EyeData with gaze and pupil
        self.data = pp.EyeData(
            left_x=np.random.uniform(500, 1400, 100),
            left_y=np.random.uniform(200, 800, 100),
            left_pupil=np.random.uniform(700, 800, 100),
            sampling_rate=100.0
        )
        # Set experimental parameters using new API
        self.data.set_experimental_setup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_screen_distance="700 mm",
            camera_spherical=("20 deg", "-90 deg", "600 mm")
        )
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_experimental_setup_3d_with_explicit_angles(self):
        """Test 3D plot with explicit theta and phi"""
        fig, ax = self.data.plot.plot_experimental_setup(
            theta="20 degrees",
            phi="-90 degrees",
            projection='3d'
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_plot_experimental_setup_3d_default_projection(self):
        """Test that 3D is the default projection"""
        fig, ax = self.data.plot.plot_experimental_setup(
            theta=np.radians(20),
            phi=np.radians(-90)
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_plot_experimental_setup_2d_with_explicit_angles(self):
        """Test 2D orthogonal projections with explicit angles"""
        fig, axes = self.data.plot.plot_experimental_setup(
            theta="20 degrees",
            phi="-90 degrees",
            projection='2d'
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 3)  # Should return 3 axes
    
    def test_plot_experimental_setup_with_pint_quantities(self):
        """Test with Pint quantities for angles"""
        fig, axes = self.data.plot.plot_experimental_setup(
            theta=20 * pp.ureg.degree,
            phi=-90 * pp.ureg.degree,
            projection='2d'
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 3)
    
    def test_plot_experimental_setup_with_calibration(self):
        """Test with ForeshorteningCalibration object"""
        from pypillometry.eyedata.foreshortening_calibration import ForeshorteningCalibration
        from pypillometry.eyedata import ExperimentalSetup
        
        # Create ExperimentalSetup for calibration
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("520 mm", "290 mm"),
            eye_screen_distance="700 mm",
            camera_spherical=("20 deg", "-90 deg", "600 mm")
        )
        
        # Create a mock calibration object
        calib = ForeshorteningCalibration(
            eye='left',
            theta=np.radians(20),
            phi=np.radians(-90),
            experimental_setup=setup,
            spline_coeffs=np.array([750.0] * 10),
            spline_knots=np.linspace(0, 10, 14),
            spline_degree=3,
            fit_intervals=None,
            fit_metrics={'R2': 0.95, 'RMSE': 10.0}
        )
        
        fig, axes = self.data.plot.plot_experimental_setup(
            calibration=calib,
            projection='2d'
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 3)
    
    def test_plot_experimental_setup_works_with_setup_angles(self):
        """Test that plot uses angles from experimental_setup when not explicitly provided"""
        # setUp already configured angles in experimental_setup
        # Plot should work without providing theta/phi explicitly
        fig, ax = self.data.plot.plot_experimental_setup(projection='3d')
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_plot_experimental_setup_no_setup_raises_error(self):
        """Test that plotting without experimental_setup raises ValueError"""
        # Create data without experimental_setup
        data = pp.EyeData(
            left_x=np.random.uniform(500, 1400, 100),
            left_y=np.random.uniform(200, 800, 100),
            left_pupil=np.random.uniform(700, 800, 100),
            sampling_rate=100.0
        )
        
        with self.assertRaises(ValueError) as context:
            data.plot.plot_experimental_setup(projection='3d')
        
        self.assertIn("experimental_setup is not set", str(context.exception))
    
    def test_plot_experimental_setup_invalid_projection_raises_error(self):
        """Test that invalid projection type raises ValueError"""
        with self.assertRaises(ValueError) as context:
            self.data.plot.plot_experimental_setup(
                theta="20 degrees",
                phi="-90 degrees",
                projection='invalid'
            )
        
        self.assertIn("projection must be", str(context.exception))
    
    def test_plot_experimental_setup_without_gaze_samples(self):
        """Test 2D plot without gaze samples"""
        fig, axes = self.data.plot.plot_experimental_setup(
            theta="20 degrees",
            phi="-90 degrees",
            projection='2d',
            show_gaze_samples=False
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 3)
    
    def test_plot_experimental_setup_custom_gaze_samples(self):
        """Test with custom number of gaze samples"""
        fig, axes = self.data.plot.plot_experimental_setup(
            theta="20 degrees",
            phi="-90 degrees",
            projection='2d',
            n_gaze_samples=16  # 4x4 grid
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 3)
    
    def test_plot_experimental_setup_custom_viewing_angle_3d(self):
        """Test 3D with custom viewing angle"""
        fig, ax = self.data.plot.plot_experimental_setup(
            theta="20 degrees",
            phi="-90 degrees",
            projection='3d',
            viewing_angle=("60 degrees", "-45 degrees")
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)


if __name__ == '__main__':
    unittest.main()

