"""
Tests for edge cases and error handling in plotting functions.
These tests verify that plotting functions handle unusual inputs gracefully.
"""
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np


class TestPlottingEdgeCases(unittest.TestCase):
    """Test edge cases for plotting functions"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_intervals_single_interval(self):
        """Test plot_intervals with Intervals object containing one interval"""
        from pypillometry.intervals import Intervals
        intervals = Intervals([(0, 1000)], units="ms", label="single")
        figs = self.data.plot.plot_intervals(intervals)
        self.assertIsNotNone(figs)
    
    def test_plot_intervals_many_intervals(self):
        """Test plot_intervals with Intervals object containing many intervals"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        figs = self.data.plot.plot_intervals(intervals, nrow=5, ncol=3)
        self.assertIsNotNone(figs)
    
    def test_pupil_plot_very_short_range(self):
        """Test pupil_plot with very short time range"""
        fig = plt.figure()
        self.data.plot.pupil_plot(plot_range=(0, 0.001), units="sec")
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_full_range(self):
        """Test pupil_plot with full data range (default)"""
        fig = plt.figure()
        self.data.plot.pupil_plot(plot_range=(-np.inf, np.inf))
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_timeseries_empty_variables(self):
        """Test plot_timeseries with empty variables list"""
        fig = plt.figure()
        # Empty list should plot all available variables
        self.data.plot.plot_timeseries(variables=[])
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_timeseries_empty_eyes(self):
        """Test plot_timeseries with empty eyes list"""
        fig = plt.figure()
        # Empty list should plot all available eyes
        self.data.plot.plot_timeseries(eyes=[])
        self.assertIsNotNone(plt.gcf())


class TestPlottingErrorHandling(unittest.TestCase):
    """Test error handling in plotting functions"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_show_onsets_invalid_option(self):
        """Test that invalid show_onsets raises ValueError"""
        fig = plt.figure()
        with self.assertRaises(ValueError):
            self.data.plot.plot_timeseries(show_onsets="invalid_option")
    
    def test_plot_intervals_invalid_type(self):
        """Test that invalid input type raises TypeError"""
        # Pass a list instead of Intervals object
        intervals = [(0, 1000), (2000, 3000)]
        with self.assertRaises(TypeError):
            self.data.plot.plot_intervals(intervals)


class TestPlottingMinimalData(unittest.TestCase):
    """Test plotting with minimal/simple data"""
    
    def setUp(self):
        """Create minimal test data"""
        self.data = pp.PupilData(
            sampling_rate=10,
            left_pupil=[1, 2, 3, 4, 5],
            right_pupil=[2, 3, 4, 5, 6]
        )
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_pupil_plot_minimal_data(self):
        """Test pupil_plot with minimal data (5 samples)"""
        fig = plt.figure()
        self.data.plot.pupil_plot()
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_timeseries_minimal_data(self):
        """Test plot_timeseries with minimal data"""
        fig = plt.figure()
        self.data.plot.plot_timeseries()
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_intervals_minimal_data(self):
        """Test plot_intervals with minimal data"""
        from pypillometry.intervals import Intervals
        intervals = Intervals([(0, 2), (2, 4)], units=None, label="minimal")
        figs = self.data.plot.plot_intervals(intervals)
        self.assertIsNotNone(figs)


class TestPlottingSingleEye(unittest.TestCase):
    """Test plotting with single eye data"""
    
    def setUp(self):
        """Create data with only left eye"""
        self.data = pp.PupilData(
            sampling_rate=100,
            left_pupil=np.sin(np.linspace(0, 10*np.pi, 1000))
        )
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_pupil_plot_single_eye(self):
        """Test pupil_plot with only one eye available"""
        fig = plt.figure()
        self.data.plot.pupil_plot()
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_request_missing_eye(self):
        """Test pupil_plot requesting non-existent eye"""
        fig = plt.figure()
        # Requesting 'right' eye which doesn't exist
        # Current implementation raises KeyError - this is expected behavior
        with self.assertRaises(KeyError):
            self.data.plot.pupil_plot(eyes=["right"])


class TestPlottingLargeData(unittest.TestCase):
    """Test plotting with large datasets"""
    
    def setUp(self):
        """Create large test data"""
        self.data = pp.PupilData(
            sampling_rate=1000,  # 1000 Hz
            left_pupil=np.random.randn(100000),  # 100 seconds
            right_pupil=np.random.randn(100000)
        )
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_pupil_plot_large_data(self):
        """Test pupil_plot with large dataset (100k samples)"""
        fig = plt.figure()
        self.data.plot.pupil_plot()
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_timeseries_large_data(self):
        """Test plot_timeseries with large dataset"""
        fig = plt.figure()
        self.data.plot.plot_timeseries()
        self.assertIsNotNone(plt.gcf())


if __name__ == '__main__':
    unittest.main()

