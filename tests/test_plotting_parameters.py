"""
Tests for plotting function parameters.
These tests verify that different parameter combinations work correctly.
"""
import unittest
import pytest
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
    
    def test_pupil_plot_show_events_false(self):
        """Test pupil_plot without events"""
        fig = plt.figure()
        self.data.plot.pupil_plot(show_events=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_pupil_plot_show_blinks_false(self):
        """Test pupil_plot without blink highlighting"""
        fig = plt.figure()
        self.data.plot.pupil_plot(show_blinks=False)
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
    
    def test_show_onsets_line(self):
        """Test plot_timeseries with line onsets"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_onsets="line")
        self.assertIsNotNone(plt.gcf())
    
    def test_show_onsets_label(self):
        """Test plot_timeseries with label onsets"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_onsets="label")
        self.assertIsNotNone(plt.gcf())
    
    def test_show_onsets_both(self):
        """Test plot_timeseries with both line and label onsets"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_onsets="both")
        self.assertIsNotNone(plt.gcf())
    
    def test_show_onsets_none(self):
        """Test plot_timeseries with no onsets"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_onsets="none")
        self.assertIsNotNone(plt.gcf())
    
    def test_show_masked_true(self):
        """Test plot_timeseries with show_masked=True (plots all data including masked regions)"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_masked=True)
        self.assertIsNotNone(plt.gcf())
    
    def test_show_masked_false(self):
        """Test plot_timeseries with show_masked=False (creates gaps at masked regions)"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_masked=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_show_mask_highlight_true(self):
        """Test plot_timeseries with mask highlighting enabled"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_mask_highlight=True)
        self.assertIsNotNone(plt.gcf())
    
    def test_show_mask_highlight_false(self):
        """Test plot_timeseries with mask highlighting disabled"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_mask_highlight=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_show_masked_false_with_highlight(self):
        """Test plot_timeseries with gaps at masked regions but still highlighting"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_masked=False, show_mask_highlight=True)
        self.assertIsNotNone(plt.gcf())
    
    def test_show_masked_true_no_highlight(self):
        """Test plot_timeseries with masked data plotted but no highlighting"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(show_masked=True, show_mask_highlight=False)
        self.assertIsNotNone(plt.gcf())
    
    def test_units_none(self):
        """Test plot_timeseries with the default units (ms)"""
        fig = plt.figure()
        self.data.plot.plot_timeseries(units="ms")
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
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms")
        
        # 2x3 layout
        figs1 = self.data.plot.plot_intervals(intervals, nrow=2, ncol=3)
        self.assertGreaterEqual(len(figs1), 1)
        
        plt.close('all')
        
        # 3x2 layout
        figs2 = self.data.plot.plot_intervals(intervals, nrow=3, ncol=2)
        self.assertGreaterEqual(len(figs2), 1)
    
    def test_plot_intervals_units(self):
        """Test plot_intervals with different units for display"""
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms")
        
        for units in ["ms", "sec", "min"]:
            intervals_in_units = intervals.to_intervals(units)
            figs = self.data.plot.plot_intervals(intervals_in_units)
            self.assertIsNotNone(figs)
            plt.close('all')
    
    def test_plot_intervals_show_mask(self):
        """Test plot_intervals with show_mask option"""
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms")
        figs = self.data.plot.plot_intervals(intervals, show_mask=True)
        self.assertIsNotNone(figs)

    def test_plot_intervals_show_mask_all_variables(self):
        """show_mask shades masks of all plotted variables, not only the last"""
        d = self.data.copy()
        for key in d.data:
            d.data.set_mask(key, np.zeros(len(d.tx)))
        intervals = d.get_epochs("F", window=(-200, 200), units="ms")
        start, end = intervals.to_intervals("samples", self.data).intervals[0]
        d.data.add_mask("left_pupil", np.s_[int(start) + 10:int(start) + 50])
        figs = d.plot.plot_intervals(intervals, eyes=["left"],
                                     variables=["pupil", "x"], show_mask=True)
        self.assertEqual(len(figs[0].axes[0].patches), 1)
        self.assertEqual(len(figs[0].axes[1].patches), 0)
    
    def test_plot_intervals_show_index(self):
        """Test plot_intervals with show_index option"""
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms")
        
        # With index
        figs1 = self.data.plot.plot_intervals(intervals, show_index=True)
        self.assertIsNotNone(figs1)
        
        plt.close('all')
        
        # Without index
        figs2 = self.data.plot.plot_intervals(intervals, show_index=False)
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
        for interv in ["0.5 min", 1, 2 * pp.u.min]:
            figs = self.data.plot.plot_timeseries_segments(interv=interv)
            self.assertIsNotNone(figs)
            self.assertGreater(len(figs), 0)
            plt.close('all')
    
    def test_segments_interv_with_units(self):
        """interv with units gives the same segments as plain numbers (bug 1.8)"""
        ref = self.data.plot.plot_timeseries_segments(interv=10, units="sec")
        n = len(ref)
        plt.close('all')
        for interv in [10 * pp.u.sec, "10 sec", 1 / 6 * pp.u.min]:
            figs = self.data.plot.plot_timeseries_segments(interv=interv, units="sec")
            self.assertEqual(len(figs), n)
            plt.close('all')

    def test_segments_with_ylim(self):
        """Test plot_timeseries_segments with ylim parameter"""
        figs = self.data.plot.plot_timeseries_segments(interv="1 min", ylim=(0, 10))
        self.assertIsNotNone(figs)

    def test_segments_ylim_all_subplots(self):
        """ylim is applied to all subplots, not only the last one"""
        figs = self.data.plot.plot_timeseries_segments(
            interv="1 min", ylim=(0, 10), variables=["pupil", "x"])
        self.assertEqual(len(figs[0].axes), 2)
        for ax in figs[0].axes:
            self.assertEqual(ax.get_ylim(), (0, 10))
        figs = self.data.plot.pupil_plot_segments(interv="1 min", ylim=(0, 10))
        for ax in figs[0].axes:
            self.assertEqual(ax.get_ylim(), (0, 10))
    
    def test_segments_different_figsize(self):
        """Test plot_timeseries_segments with different figure sizes"""
        figs = self.data.plot.plot_timeseries_segments(
            interv="1 min", 
            figsize=(20, 10)
        )
        self.assertIsNotNone(figs)
    
    def test_segments_with_eyes(self):
        """Test plot_timeseries_segments with specific eyes"""
        figs = self.data.plot.plot_timeseries_segments(
            interv="1 min",
            eyes=["left"]
        )
        self.assertIsNotNone(figs)
    
    def test_segments_with_variables(self):
        """Test plot_timeseries_segments with specific variables"""
        # Get available variables
        available_vars = list(set([k.split('_')[1] for k in self.data.data.keys() if '_' in k]))
        if len(available_vars) > 0:
            figs = self.data.plot.plot_timeseries_segments(
                interv="1 min",
                variables=[available_vars[0]]
            )
            self.assertIsNotNone(figs)
    
    def test_segments_with_units(self):
        """Test plot_timeseries_segments with different units"""
        # Use appropriate interval values for each unit
        units_intervals = [("sec", "10 sec"), ("ms", 10000.0), ("min", "1 min")]
        for units, interv in units_intervals:
            figs = self.data.plot.plot_timeseries_segments(
                interv=interv,
                units=units
            )
            self.assertIsNotNone(figs)
            plt.close('all')


if __name__ == '__main__':
    unittest.main()



def test_plot_range_includes_end_sample():
    # the sample at the end of plot_range was left out
    d = pp.EyeData(left_pupil=np.arange(100.), left_x=np.arange(100.),
                   left_y=np.arange(100.), sampling_rate=100)
    try:
        for plot in (d.plot.plot_timeseries, d.plot.pupil_plot):
            plt.figure()
            plot(plot_range=(100, 500), units="ms")
            xs = [line.get_xdata() for ax in plt.gcf().axes for line in ax.lines
                  if len(line.get_xdata()) > 2]
            assert xs and all(x[0] == 100 and x[-1] == 500 for x in xs), plot
            plt.close("all")
    finally:
        plt.close("all")


@pytest.mark.parametrize("plot_range", [(1, 3), (1 * pp.u.sec, 3 * pp.u.sec), ("1000 ms", 3)])
def test_plot_range_plain_numbers_in_units(plot_range):
    # plain numbers in plot_range are in `units` (2.1 scripts: units="sec")
    d = pp.EyeData(left_pupil=np.arange(1000.), left_x=np.arange(1000.),
                   left_y=np.arange(1000.), sampling_rate=100)
    try:
        for plot in (d.plot.plot_timeseries, d.plot.pupil_plot):
            plt.figure()
            plot(plot_range=plot_range, units="sec")
            xs = [line.get_xdata() for ax in plt.gcf().axes for line in ax.lines
                  if len(line.get_xdata()) > 2]
            assert xs and all(x[0] == 1 and x[-1] == 3 for x in xs), plot
            plt.close("all")
    finally:
        plt.close("all")


def test_default_time_units_are_seconds():
    d = pp.EyeData(left_pupil=np.arange(1000.), left_x=np.arange(1000.),
                   left_y=np.arange(1000.), sampling_rate=100,
                   event_onsets=[1000., 5000.], event_labels=["a", "b"])
    d.mask_intervals([(1000, 2000)])  # plot_masked needs a masked period
    try:
        for plot in (d.plot.plot_timeseries, d.plot.pupil_plot, d.plot.plot_masked,
                     d.events.plot):
            plt.figure()
            plot()
            assert any("sec" in ax.get_xlabel() for ax in plt.gcf().axes), plot
            plt.close("all")
        figs = d.plot.plot_timeseries_segments(interv=5)  # 5 sec
        assert len(figs) == 2 and "sec" in [ax.get_xlabel() for ax in figs[0].axes]
    finally:
        plt.close("all")


class TestHighlight(unittest.TestCase):
    """highlight= and highlight_color= of the plots with a time axis"""

    def setUp(self):
        tx = np.arange(0, 10000, 10.)
        self.d = pp.EyeData(time=tx, left_pupil=np.ones_like(tx), left_x=np.ones_like(tx),
                            left_y=np.ones_like(tx), event_onsets=[1000, 6000],
                            event_labels=["S_1", "S_2"])
        self.iv = pp.Intervals([(1000, 2000), (9500, 12000)], units="ms")
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def spans(ax):
        """(start, end, color) of the shaded spans of `ax`."""
        out = []
        for p in ax.patches:
            x = p.get_xy()[:, 0] if hasattr(p, "get_xy") and np.ndim(p.get_xy()) == 2 \
                else [p.get_x(), p.get_x() + p.get_width()]
            out.append((round(float(min(x)), 3), round(float(max(x)), 3),
                        matplotlib.colors.to_hex(p.get_facecolor())))
        return sorted(out)

    def test_pupil_plot_converts_units_and_clips(self):
        plt.figure()
        self.d.plot.pupil_plot(units="sec", highlight=self.iv, highlight_color="green")
        # in seconds; the second interval is cut at the end of the data
        self.assertEqual(self.spans(plt.gca()),
                         [(1.0, 2.0, "#008000"), (9.5, 9.99, "#008000")])

    def test_several_with_colors(self):
        plt.figure()
        ep = self.d.get_epochs("S_", (0, 500))
        self.d.plot.pupil_plot(units="ms", highlight=[self.iv, ep],
                               highlight_color=("red", "blue"))
        spans = self.spans(plt.gca())
        self.assertIn((1000.0, 1500.0, "#0000ff"), spans)
        self.assertIn((6000.0, 6500.0, "#0000ff"), spans)
        self.assertIn((1000.0, 2000.0, "#ff0000"), spans)

    def test_one_color_for_all_and_rgb_tuple(self):
        plt.figure()
        self.d.plot.pupil_plot(units="ms", highlight=(self.iv, self.iv),
                               highlight_color=(0.0, 0.5, 0.0))
        self.assertEqual(len(self.spans(plt.gca())), 4)

    def test_plot_timeseries_all_subplots(self):
        plt.figure()
        self.d.plot.plot_timeseries(units="ms", highlight=self.iv)
        axs = plt.gcf().axes
        self.assertEqual(len(axs), 3)
        for ax in axs:
            self.assertIn((1000.0, 2000.0, "#add8e6"), self.spans(ax))

    def test_plot_range_limits_spans(self):
        plt.figure()
        self.d.plot.plot_timeseries(units="ms", plot_range=(1500, 3000),
                                    variables="pupil", highlight=self.iv)
        self.assertEqual(self.spans(plt.gca()), [(1500.0, 2000.0, "#add8e6")])

    def test_samples_and_plot_masked(self):
        iv = pp.Intervals([(100, 200)], units="samples")
        d = self.d.mask_intervals([(4000, 4500)], inplace=False)
        plt.figure()
        d.plot.plot_masked(units="sec", highlight=iv, show_labels=False)
        self.assertEqual(self.spans(plt.gca()), [(1.0, 2.0, "#add8e6")])

    def test_segments_pass_highlight_on(self):
        figs = self.d.plot.plot_timeseries_segments(interv="5 sec", variables="pupil",
                                                    highlight=self.iv)
        self.assertEqual(self.spans(figs[0].axes[0]), [(1.0, 2.0, "#add8e6")])

    def test_errors(self):
        plt.figure()
        with self.assertRaisesRegex(TypeError, "Intervals or Epochs"):
            self.d.plot.pupil_plot(highlight=[(1000, 2000)])
        with self.assertRaisesRegex(ValueError, "2 highlight colors for 1"):
            self.d.plot.pupil_plot(highlight=self.iv, highlight_color=["red", "blue"])
