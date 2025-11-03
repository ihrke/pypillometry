"""
Tests for plotting function parameters.
These tests verify that different parameter combinations work correctly.
"""
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np


class TestPupilPlotParameters(unittest.TestCase):
    """Test different parameter combinations for pupil_plot"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_pupil_plot_units_sec(self):
        """Test pupil_plot with seconds"""
        fig = plt.figure()
        self.data.plot.pupil_plot(units="sec")
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_units_ms(self):
        """Test pupil_plot with milliseconds"""
        fig = plt.figure()
        self.data.plot.pupil_plot(units="ms")
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_units_min(self):
        """Test pupil_plot with minutes"""
        fig = plt.figure()
        self.data.plot.pupil_plot(units="min")
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_units_h(self):
        """Test pupil_plot with hours"""
        fig = plt.figure()
        self.data.plot.pupil_plot(units="h")
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_eyes_empty_list(self):
        """Test pupil_plot with empty eyes list (should plot all)"""
        fig = plt.figure()
        self.data.plot.pupil_plot(eyes=[])
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_eyes_left(self):
        """Test pupil_plot with left eye only"""
        fig = plt.figure()
        self.data.plot.pupil_plot(eyes=["left"])
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_eyes_right(self):
        """Test pupil_plot with right eye only"""
        fig = plt.figure()
        self.data.plot.pupil_plot(eyes=["right"])
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_eyes_string(self):
        """Test pupil_plot with single eye as string (not list)"""
        fig = plt.figure()
        self.data.plot.pupil_plot(eyes="left")
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_plot_events_false(self):
        """Test pupil_plot without events"""
        fig = plt.figure()
        self.data.plot.pupil_plot(plot_events=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_highlight_blinks_false(self):
        """Test pupil_plot without blink highlighting"""
        fig = plt.figure()
        self.data.plot.pupil_plot(highlight_blinks=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_style_single_dict(self):
        """Test pupil_plot with single style dict (should auto-differentiate eyes)"""
        fig = plt.figure()
        self.data.plot.pupil_plot(eyes=["left", "right"], style={'color': 'red', 'alpha': 0.7})
        self.assertIsNotNone(plt.gcf())
        # Check that lines were created
        lines = plt.gca().get_lines()
        self.assertGreater(len(lines), 0)
    
    def test_pupil_plot_style_per_eye(self):
        """Test pupil_plot with per-eye style dict"""
        fig = plt.figure()
        self.data.plot.pupil_plot(
            eyes=["left", "right"],
            style={
                'left': {'color': 'blue', 'linestyle': '-'},
                'right': {'color': 'green', 'linestyle': '--'}
            }
        )
        self.assertIsNotNone(plt.gcf())
        lines = plt.gca().get_lines()
        self.assertEqual(len(lines), 2)


class TestPlotTimeseriesParameters(unittest.TestCase):
    """Test different parameter combinations for plot_timeseries"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_onsets_line(self):
        """Test plot_timeseries with line onsets"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(plot_onsets="line")
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_onsets_label(self):
        """Test plot_timeseries with label onsets"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(plot_onsets="label")
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_onsets_both(self):
        """Test plot_timeseries with both line and label onsets"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(plot_onsets="both")
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_onsets_none(self):
        """Test plot_timeseries with no onsets"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(plot_onsets="none")
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_masked_true(self):
        """Test plot_timeseries with masked regions highlighted"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(plot_masked=True)
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_masked_false(self):
        """Test plot_timeseries without masked regions"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(plot_masked=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_units_none(self):
        """Test plot_timeseries with units=None"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(units=None)
        self.assertIsNotNone(plt.gcf())
    
    def test_plot_timeseries_style_single_dict(self):
        """Test plot_timeseries with single style dict (should auto-differentiate eyes)"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(eyes=["left", "right"], style={'color': 'blue', 'alpha': 0.8})
        self.assertIsNotNone(plt.gcf())
        # Check that axes were created
        self.assertGreater(len(fig.axes), 0)
    
    def test_plot_timeseries_style_per_eye(self):
        """Test plot_timeseries with per-eye style dict"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(
            eyes=["left", "right"],
            style={
                'left': {'color': 'red', 'linestyle': '-'},
                'right': {'color': 'purple', 'linestyle': ':'}
            }
        )
        self.assertIsNotNone(plt.gcf())
        self.assertGreater(len(fig.axes), 0)


class TestPlotIntervalsParameters(unittest.TestCase):
    """Test different parameter combinations for plot_intervals"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_intervals_different_layouts(self):
        """Test plot_intervals with different nrow/ncol combinations using Intervals"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        # 2x3 layout
        figs1 = self.data.plot.plot_intervals(intervals, nrow=2, ncol=3)
        self.assertGreaterEqual(len(figs1), 1)
        
        plt.close('all')
        
        # 3x2 layout
        figs2 = self.data.plot.plot_intervals(intervals, nrow=3, ncol=2)
        self.assertGreaterEqual(len(figs2), 1)
    
    def test_plot_intervals_units(self):
        """Test plot_intervals with different units for display"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        for units in ["ms", "sec", "min"]:
            figs = self.data.plot.plot_intervals(intervals, units=units)
            self.assertIsNotNone(figs)
            plt.close('all')
    
    def test_plot_intervals_plot_mask(self):
        """Test plot_intervals with plot_mask option"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        figs = self.data.plot.plot_intervals(intervals, plot_mask=True)
        self.assertIsNotNone(figs)
    
    def test_plot_intervals_plot_index(self):
        """Test plot_intervals with plot_index option"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        # With index
        figs1 = self.data.plot.plot_intervals(intervals, plot_index=True)
        self.assertIsNotNone(figs1)
        
        plt.close('all')
        
        # Without index
        figs2 = self.data.plot.plot_intervals(intervals, plot_index=False)
        self.assertIsNotNone(figs2)


class TestSegmentPlotParameters(unittest.TestCase):
    """Test different parameter combinations for segment plotting"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_segments_different_intervals(self):
        """Test plot_timeseries_segments with different interval sizes"""
        for interv in [0.5, 1.0, 2.0]:
            figs = self.data.plot.plot_timeseries_segments(interv=interv)
            self.assertIsNotNone(figs)
            self.assertGreater(len(figs), 0)
            plt.close('all')
    
    def test_segments_with_ylim(self):
        """Test plot_timeseries_segments with ylim parameter"""
        figs = self.data.plot.plot_timeseries_segments(interv=1.0, ylim=(0, 10))
        self.assertIsNotNone(figs)
    
    def test_segments_different_figsize(self):
        """Test plot_timeseries_segments with different figure sizes"""
        figs = self.data.plot.plot_timeseries_segments(
            interv=1.0, 
            figsize=(20, 10)
        )
        self.assertIsNotNone(figs)
    
    def test_segments_with_eyes(self):
        """Test plot_timeseries_segments with specific eyes"""
        figs = self.data.plot.plot_timeseries_segments(
            interv=1.0,
            eyes=["left"]
        )
        self.assertIsNotNone(figs)
    
    def test_segments_with_variables(self):
        """Test plot_timeseries_segments with specific variables"""
        # Get available variables
        available_vars = list(set([k.split('_')[1] for k in self.data.data.keys() if '_' in k]))
        if len(available_vars) > 0:
            figs = self.data.plot.plot_timeseries_segments(
                interv=1.0,
                variables=[available_vars[0]]
            )
            self.assertIsNotNone(figs)
    
    def test_segments_with_units(self):
        """Test plot_timeseries_segments with different units"""
        # Use appropriate interval values for each unit
        units_intervals = [("sec", 1.0), ("ms", 1000.0), ("min", 1.0)]
        for units, interv in units_intervals:
            figs = self.data.plot.plot_timeseries_segments(
                interv=interv,
                units=units
            )
            self.assertIsNotNone(figs)
            plt.close('all')


if __name__ == '__main__':
    unittest.main()

