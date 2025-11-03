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


if __name__ == '__main__':
    unittest.main()

