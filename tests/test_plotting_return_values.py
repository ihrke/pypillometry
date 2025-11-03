"""
Tests for plotting function return values.
These tests verify that plotting functions return the expected objects and structures.
"""
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np


class TestPlottingReturnValues(unittest.TestCase):
    """Test that plotting functions return correct types and structures"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_intervals_returns_list(self):
        """Test that plot_intervals returns a list"""
        intervals = [(0, 1000), (2000, 3000)]
        figs = self.data.plot.plot_intervals(intervals, nrow=1, ncol=1)
        self.assertIsInstance(figs, list)
    
    def test_plot_intervals_returns_figures(self):
        """Test that plot_intervals returns Figure objects"""
        intervals = [(0, 1000), (2000, 3000)]
        figs = self.data.plot.plot_intervals(intervals, nrow=1, ncol=1)
        for fig in figs:
            self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_intervals_correct_figure_count(self):
        """Test that plot_intervals returns correct number of figures"""
        # With nrow=2, ncol=2, we have 4 subplots per figure
        # 5 intervals should give us 2 figures
        intervals = [(i*1000, (i+1)*1000) for i in range(5)]
        figs = self.data.plot.plot_intervals(intervals, nrow=2, ncol=2)
        self.assertEqual(len(figs), 2)
    
    def test_plot_timeseries_creates_axes(self):
        """Test that plot_timeseries creates the correct number of axes"""
        fig = plt.figure()
        # Get available variables
        available_vars = list(set([k.split('_')[1] for k in self.data.data.keys() if '_' in k]))
        if len(available_vars) > 0:
            self.data.plot.plot_timeseries(variables=[available_vars[0]])
            # Should create 1 subplot for 1 variable
            self.assertGreaterEqual(len(fig.axes), 1)
    
    def test_plot_timeseries_segments_returns_list(self):
        """Test that plot_timeseries_segments returns a list"""
        figs = self.data.plot.plot_timeseries_segments(interv=1.0)
        self.assertIsInstance(figs, list)
    
    def test_plot_timeseries_segments_returns_figures(self):
        """Test that plot_timeseries_segments returns Figure objects"""
        figs = self.data.plot.plot_timeseries_segments(interv=1.0)
        for fig in figs:
            self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_timeseries_segments_figure_count(self):
        """Test that correct number of figures are created"""
        # Calculate expected number based on data duration
        duration_min = self.data.tx.max() / 1000 / 60
        interv = 1.0  # minutes
        expected_figs = int(np.ceil(duration_min / interv))
        
        figs = self.data.plot.plot_timeseries_segments(interv=interv)
        self.assertEqual(len(figs), expected_figs)
    
    def test_pupil_plot_segments_returns_list(self):
        """Test that pupil_plot_segments returns a list"""
        figs = self.data.plot.pupil_plot_segments(interv=1.0)
        self.assertIsInstance(figs, list)
    
    def test_pupil_plot_segments_returns_figures(self):
        """Test that pupil_plot_segments returns Figure objects"""
        figs = self.data.plot.pupil_plot_segments(interv=1.0)
        for fig in figs:
            self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_blinks_returns_list(self):
        """Test that plot_blinks returns a list"""
        figs = self.data.plot.plot_blinks()
        self.assertIsInstance(figs, list)
    
    def test_plot_blinks_returns_figures(self):
        """Test that plot_blinks returns Figure objects"""
        figs = self.data.plot.plot_blinks()
        for fig in figs:
            self.assertIsInstance(fig, plt.Figure)


class TestPlottingEmptyReturns(unittest.TestCase):
    """Test return values for edge cases"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_intervals_empty_list(self):
        """Test with empty intervals list"""
        figs = self.data.plot.plot_intervals([])
        self.assertIsInstance(figs, list)
        self.assertEqual(len(figs), 0)


if __name__ == '__main__':
    unittest.main()

