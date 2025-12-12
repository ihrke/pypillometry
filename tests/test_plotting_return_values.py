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
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        figs = self.data.plot.plot_intervals(intervals, nrow=1, ncol=1)
        self.assertIsInstance(figs, list)
    
    def test_plot_intervals_returns_figures(self):
        """Test that plot_intervals returns Figure objects"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        figs = self.data.plot.plot_intervals(intervals, nrow=1, ncol=1)
        for fig in figs:
            self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_intervals_correct_figure_count(self):
        """Test that plot_intervals returns correct number of figures"""
        # With nrow=2, ncol=2, we have 4 subplots per figure
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        if len(intervals) >= 5:
            # Take first 5 intervals
            from pypillometry.intervals import Intervals
            intervals_subset = Intervals(
                intervals.intervals[:5],
                intervals.units,
                intervals.label,
                event_labels=intervals.event_labels[:5] if intervals.event_labels else None,
                event_indices=intervals.event_indices[:5] if intervals.event_indices is not None else None,
                sampling_rate=intervals.sampling_rate
            )
            figs = self.data.plot.plot_intervals(intervals_subset, nrow=2, ncol=2)
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
        # Duration is the difference between max and min time, not just max time
        duration_min = (self.data.tx.max() - self.data.tx.min()) / 1000 / 60
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
    
    def test_plot_intervals_with_intervals_object(self):
        """Test with Intervals object"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        figs = self.data.plot.plot_intervals(intervals)
        self.assertIsInstance(figs, list)
        self.assertGreater(len(figs), 0)


if __name__ == '__main__':
    unittest.main()

