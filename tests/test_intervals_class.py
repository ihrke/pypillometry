"""
Tests for the Intervals class.
"""
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np
import pandas as pd
from pypillometry.intervals import Intervals, concat_intervals
from types import SimpleNamespace


def timebase(n, fs=1000., t0=0.):
    """Minimal data object (time vector in ms, sampling rate) for conversions."""
    return SimpleNamespace(tx=t0 + np.arange(n) * 1000. / fs, fs=fs)


def spans(d, event, window, units):
    """The spans of ``d.get_epochs(event, window)`` as Intervals in `units`."""
    return d.get_epochs(event, window=window, units=units).to_intervals(units, d)


class TestIntervalsClass(unittest.TestCase):
    """Test Intervals class basic functionality"""
    
    def test_intervals_creation_from_list(self):
        """Test creating Intervals from list of tuples"""
        intervals_list = [(0, 100), (200, 300), (400, 500)]
        intervals = Intervals(intervals_list, units="ms", label="test")
        
        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals.units, "ms")
        self.assertEqual(intervals.label, "test")
    
    def test_intervals_creation_from_array(self):
        """Test creating Intervals from numpy array"""
        intervals_array = np.array([[0, 100], [200, 300], [400, 500]])
        intervals = Intervals(intervals_array, units="sec")
        
        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals.units, "sec")
    
    def test_intervals_iteration(self):
        """Test iterating over Intervals"""
        intervals_list = [(0, 100), (200, 300)]
        intervals = Intervals(intervals_list, units="ms")
        
        result = list(intervals)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (0, 100))
        self.assertEqual(result[1], (200, 300))
    
    def test_intervals_indexing(self):
        """Test indexing Intervals"""
        intervals_list = [(0, 100), (200, 300), (400, 500)]
        intervals = Intervals(intervals_list, units="ms")
        
        self.assertEqual(intervals[0], (0, 100))
        self.assertEqual(intervals[1], (200, 300))
        self.assertEqual(intervals[-1], (400, 500))
    
    def test_intervals_to_array(self):
        """Test converting Intervals to array"""
        intervals_list = [(0, 100), (200, 300)]
        intervals = Intervals(intervals_list, units="ms")
        
        arr = np.array(intervals)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (2, 2))
        np.testing.assert_array_equal(arr, [[0, 100], [200, 300]])
    
    def test_intervals_to_array_dtype(self):
        """np.asarray with dtype (numpy 2 passes dtype and copy)"""
        intervals = Intervals([(0.5, 100.2), (200., 300.)], units="ms")
        arr = np.asarray(intervals, dtype=int)
        self.assertEqual(arr.dtype, int)
        np.testing.assert_array_equal(arr, [[0, 100], [200, 300]])
        np.testing.assert_array_equal(np.array(intervals, copy=True), [[0.5, 100.2], [200., 300.]])

    def test_intervals_with_metadata(self):
        """Test Intervals with event metadata"""
        intervals_list = [(0, 100), (200, 300)]
        event_labels = ["event1", "event2"]
        
        intervals = Intervals(
            intervals_list, 
            units="ms", 
            label="stimuli",
            event_labels=event_labels,
        )
        
        self.assertEqual(intervals.label, "stimuli")
        self.assertEqual(intervals.event_labels, ["event1", "event2"])

    def test_intervals_as_pandas(self):
        """Test conversion of intervals to pandas DataFrame"""
        intervals_list = [(0, 100), (200, 260)]
        event_labels = ["stim", "resp"]
        event_onsets = [0, 200]
        intervals = Intervals(
            intervals_list,
            units="ms",
            label="events",
            event_labels=event_labels,
            event_onsets=event_onsets,
        )

        df = intervals.as_pandas()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertTrue((df["start"].values == np.array([0, 200])).all())
        self.assertTrue((df["duration"].values == np.array([100, 60])).all())
        self.assertTrue((df["label"].values == np.array(event_labels)).all())
        self.assertTrue((df["onset"].values == np.array(event_onsets)).all())

        empty_df = Intervals([], units="ms").as_pandas()
        self.assertTrue(empty_df.empty)


class TestIntervalsMethods(unittest.TestCase):
    """Test Intervals class methods"""
    
    def test_intervals_merge(self):
        """Test merging overlapping intervals"""
        # Overlapping intervals
        intervals_list = [(0, 100), (50, 150), (200, 300)]
        intervals = Intervals(intervals_list, units="ms", label="test")
        
        merged = intervals.merge()
        self.assertIsInstance(merged, Intervals)
        self.assertEqual(len(merged), 2)  # Should merge first two
        self.assertEqual(merged.units, "ms")
        self.assertEqual(merged.label, "test")
    
    def test_intervals_merge_with_metadata(self):
        """Test that merge preserves and combines metadata"""
        intervals_list = [(0, 100), (50, 150), (200, 300)]
        event_labels = ["event1", "event2", "event3"]
        event_onsets = [10, 60, 210]
        
        intervals = Intervals(intervals_list, units="ms", label="test",
                             event_labels=event_labels,
                             event_onsets=event_onsets)
        
        merged = intervals.merge()
        
        # Should have 2 intervals (first two merged, third separate)
        self.assertEqual(len(merged), 2)
        
        # First interval: merged labels with underscore
        self.assertEqual(merged.event_labels[0], "event1_event2")
        # Second interval: unchanged label
        self.assertEqual(merged.event_labels[1], "event3")
        
        # First interval: first index from merged group
        # Second interval: original index
        
        # First interval: first onset from merged group
        self.assertEqual(merged.event_onsets[0], 10)
        # Second interval: original onset
        self.assertEqual(merged.event_onsets[1], 210)
    
    def test_intervals_merge_custom_separator(self):
        """Test merge with custom separator for labels"""
        intervals_list = [(0, 100), (50, 150)]
        event_labels = ["a", "b"]
        
        intervals = Intervals(intervals_list, units="ms",
                             event_labels=event_labels)
        
        merged = intervals.merge(merge_sep="|")
        
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.event_labels[0], "a|b")
    
    def test_intervals_merge_empty_returns_self(self):
        """Test that merging empty intervals returns self"""
        intervals = Intervals([], units="ms")
        
        merged = intervals.merge()
        
        self.assertIs(merged, intervals)
    
    def test_intervals_merge_with_distance_ms(self):
        """Test merging intervals with distance parameter in ms units"""
        # Gap between (10,15) and (17,22) is 2ms
        intervals = Intervals([(10, 15), (17, 22), (30, 35)], units="ms")
        
        # distance=0: no merging of non-overlapping
        merged_0 = intervals.merge(distance=0)
        self.assertEqual(len(merged_0), 3)
        
        # distance=2: should merge first two (gap is exactly 2)
        merged_2 = intervals.merge(distance=2)
        self.assertEqual(len(merged_2), 2)
        self.assertEqual(merged_2.intervals[0], (10, 22))
        self.assertEqual(merged_2.intervals[1], (30, 35))
        
        # distance=1: should NOT merge (gap is 2 > 1)
        merged_1 = intervals.merge(distance=1)
        self.assertEqual(len(merged_1), 3)
        
        # distance=20: should merge all three
        merged_20 = intervals.merge(distance=20)
        self.assertEqual(len(merged_20), 1)
        self.assertEqual(merged_20.intervals[0], (10, 35))
    
    def test_intervals_merge_with_distance_indices(self):
        """distance is in the units of the intervals (here samples)"""
        intervals = Intervals([(10, 15), (17, 22), (30, 35)], units="samples")
        merged = intervals.merge(distance=2)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.intervals[0], (10, 22))
        self.assertEqual(intervals.merge(distance=2 * pp.u.samples).intervals,
                         merged.intervals)
        with self.assertRaises(ValueError):
            intervals.merge(distance=2 * pp.u.ms)

    def test_intervals_merge_with_distance_seconds(self):
        """distance is in the units of the intervals unless it carries units"""
        intervals = Intervals([(0.010, 0.015), (0.017, 0.022)], units="sec")
        self.assertEqual(len(intervals.merge(distance=0.001)), 2)
        for distance in [0.002, 2 * pp.u.ms, "2 ms"]:
            merged = intervals.merge(distance=distance)
            self.assertEqual(len(merged), 1)
            self.assertAlmostEqual(merged.intervals[0][0], 0.010)
            self.assertAlmostEqual(merged.intervals[0][1], 0.022)

    def test_intervals_merge_with_distance_preserves_metadata(self):
        """Test that merge with distance preserves and combines metadata correctly"""
        intervals = Intervals(
            [(10, 15), (17, 22), (30, 35)], 
            units="ms",
            event_labels=["a", "b", "c"],
            event_onsets=[10, 17, 30]
        )
        
        # Merge first two intervals
        merged = intervals.merge(distance=2)
        
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.event_labels[0], "a_b")  # Labels joined
        self.assertEqual(merged.event_labels[1], "c")
        self.assertEqual(merged.event_onsets[0], 10)  # First onset kept
    
    def test_intervals_merge_with_distance_preserves_units(self):
        """Test that merge with distance returns intervals in original units"""
        for units in ["ms", "sec", "min"]:
            intervals = Intervals([(10, 15), (17, 22)], units=units)
            merged = intervals.merge(distance=1000)  # Large distance to force merge
            self.assertEqual(merged.units, units)
    
    def test_intervals_pad_basic(self):
        """Test basic padding of intervals"""
        intervals = Intervals([(100, 200), (300, 400)], units="ms")
        
        padded = intervals.pad(left=10, right=20)
        
        self.assertEqual(len(padded), 2)
        self.assertEqual(padded.intervals[0], (90, 220))
        self.assertEqual(padded.intervals[1], (290, 420))
    
    def test_intervals_pad_left_only(self):
        """Test padding only on the left"""
        intervals = Intervals([(100, 200)], units="ms")
        
        padded = intervals.pad(left=50)
        
        self.assertEqual(padded.intervals[0], (50, 200))
    
    def test_intervals_pad_right_only(self):
        """Test padding only on the right"""
        intervals = Intervals([(100, 200)], units="ms")
        
        padded = intervals.pad(right=50)
        
        self.assertEqual(padded.intervals[0], (100, 250))
    
    def test_intervals_pad_preserves_metadata(self):
        """Test that pad preserves all metadata"""
        intervals = Intervals(
            [(100, 200), (300, 400)],
            units="ms",
            label="test",
            event_labels=["a", "b"],
            event_onsets=[100, 300]
        )
        
        padded = intervals.pad(left=10, right=10)
        
        self.assertEqual(padded.units, "ms")
        self.assertEqual(padded.label, "test")
        self.assertEqual(padded.event_labels, ["a", "b"])
        self.assertEqual(padded.event_onsets, [100, 300])
    
    def test_intervals_pad_empty_returns_self(self):
        """Test that padding empty intervals returns self"""
        intervals = Intervals([], units="ms")
        
        padded = intervals.pad(left=10, right=10)
        
        self.assertIs(padded, intervals)
    
    def test_intervals_pad_with_indices(self):
        """Test padding with index units"""
        intervals = Intervals([(10, 20), (30, 40)], units="samples")
        
        padded = intervals.pad(left=5, right=5)
        
        self.assertEqual(padded.intervals[0], (5, 25))
        self.assertEqual(padded.intervals[1], (25, 45))
    
    def test_intervals_pad_then_merge(self):
        """Test chaining pad with merge"""
        # Two intervals with a 10-unit gap
        intervals = Intervals([(100, 150), (160, 200)], units="ms")
        
        # Padding by 10 on each side should make them overlap
        padded = intervals.pad(left=10, right=10)
        self.assertEqual(padded.intervals[0], (90, 160))
        self.assertEqual(padded.intervals[1], (150, 210))
        
        # Merging should combine them
        merged = padded.merge()
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.intervals[0], (90, 210))
    
    def test_intervals_pad_negative_shrinks(self):
        """Test that negative padding shrinks intervals"""
        intervals = Intervals([(100, 200)], units="ms")
        
        # Negative left adds to start, negative right subtracts from end
        padded = intervals.pad(left=-10, right=-20)
        
        self.assertEqual(padded.intervals[0], (110, 180))
    
    def test_intervals_pad_not_clipped(self):
        """padding is not clipped (time intervals, or sample indices before 0)"""
        intervals = Intervals([(10, 50)], units="ms")
        self.assertEqual(intervals.pad(left=20).intervals[0], (-10, 50))
        intervals = Intervals([(5, 20)], units="samples")
        self.assertEqual(intervals.pad(left=10).intervals[0], (-5, 20))
    
    def test_intervals_stats(self):
        """Test getting interval statistics"""
        intervals_list = [(0, 100), (0, 200), (0, 300)]
        intervals = Intervals(intervals_list, units="ms")
        
        stats = intervals.stats()
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["mean"], 200.0)  # (100+200+300)/3
        self.assertEqual(stats["min"], 100.0)
        self.assertEqual(stats["max"], 300.0)

    def test_intervals_stats_empty(self):
        """stats() of empty Intervals gives n=0 and NaN"""
        stats = Intervals([], units="ms").stats()
        self.assertEqual(stats["n"], 0)
        self.assertEqual(stats["total_duration"], 0)
        for key in ("mean", "sd", "min", "max"):
            self.assertTrue(np.isnan(stats[key]))
    
    def test_intervals_repr(self):
        """Test string representation of Intervals"""
        intervals_list = [(0, 100), (0, 200), (0, 300)]
        intervals = Intervals(intervals_list, units="ms", label="test_intervals")
        
        repr_str = repr(intervals)
        self.assertIn("test_intervals", repr_str)
        self.assertIn("3 intervals", repr_str)
        self.assertIn("ms", repr_str)  # units shown inline with values

    def test_intervals_repr_indices(self):
        """Index intervals show 'samples' as unit, not 'None'"""
        repr_str = repr(Intervals([(0, 10), (20, 30)], units="samples"))
        self.assertIn("samples", repr_str)
        self.assertNotIn("None", repr_str)
    
    def test_intervals_repr_empty(self):
        """Test string representation of empty Intervals"""
        intervals = Intervals([], units="ms", label="empty")
        
        repr_str = repr(intervals)
        self.assertIn("empty", repr_str)
        self.assertIn("0 intervals", repr_str)
    
    def test_intervals_repr_no_label(self):
        """Test string representation without label"""
        intervals_list = [(0, 100)]
        intervals = Intervals(intervals_list, units="samples")
        
        repr_str = repr(intervals)
        self.assertIn("Intervals", repr_str)
        self.assertIn("1 intervals", repr_str)  # check interval count instead of units format


class TestIntervalsAddOperator(unittest.TestCase):
    """Test Intervals + operator for combining intervals"""
    
    def test_add_same_units(self):
        """Test adding two Intervals with same units"""
        a = Intervals([(0, 100), (200, 300)], units="ms", label="a")
        b = Intervals([(500, 600)], units="ms", label="b")
        c = a + b
        
        self.assertEqual(len(c), 3)
        self.assertEqual(c.intervals, [(0, 100), (200, 300), (500, 600)])
        self.assertEqual(c.units, "ms")
        self.assertEqual(c.label, "a + b")
    
    def test_add_different_units_converts(self):
        """Test that adding with different units converts to left operand's units"""
        a = Intervals([(0, 1), (2, 3)], units="sec")
        b = Intervals([(4000, 5000)], units="ms")
        c = a + b
        
        self.assertEqual(len(c), 3)
        self.assertEqual(c.units, "sec")
        # 4000ms = 4sec, 5000ms = 5sec
        self.assertAlmostEqual(c.intervals[2][0], 4.0)
        self.assertAlmostEqual(c.intervals[2][1], 5.0)
    
    def test_add_preserves_event_labels(self):
        """Test that event_labels are concatenated"""
        a = Intervals([(0, 100)], units="ms", event_labels=["event1"])
        b = Intervals([(200, 300)], units="ms", event_labels=["event2"])
        c = a + b
        
        self.assertEqual(c.event_labels, ["event1", "event2"])
    
    def test_add_preserves_event_onsets(self):
        """Test that event_onsets are concatenated and converted"""
        a = Intervals([(0, 100)], units="ms", event_onsets=np.array([50]))
        b = Intervals([(200, 300)], units="ms", event_onsets=np.array([250]))
        c = a + b
        
        np.testing.assert_array_equal(c.event_onsets, [50, 250])
    
    def test_add_partial_metadata_left_only(self):
        """Test adding when only left operand has metadata"""
        a = Intervals([(0, 100)], units="ms", event_labels=["a"])
        b = Intervals([(200, 300)], units="ms")
        c = a + b
        
        # one label per interval ("" where it is missing)
        self.assertEqual(c.event_labels, ["a", ""])
    
    def test_add_partial_metadata_right_only(self):
        """Test adding when only right operand has metadata"""
        a = Intervals([(0, 100)], units="ms")
        b = Intervals([(200, 300)], units="ms", event_labels=["b"])
        c = a + b
        
        self.assertEqual(c.event_labels, ["", "b"])
    
    def test_add_no_metadata(self):
        """Test adding when neither operand has metadata"""
        a = Intervals([(0, 100)], units="ms")
        b = Intervals([(200, 300)], units="ms")
        c = a + b
        
        self.assertIsNone(c.event_labels)
        self.assertIsNone(c.event_onsets)
    
    def test_add_indices_units(self):
        """Test adding intervals with units="samples" (indices)"""
        a = Intervals([(0, 100), (200, 300)], units="samples")
        b = Intervals([(500, 600)], units="samples")
        c = a + b
        
        self.assertEqual(len(c), 3)
        self.assertEqual(c.units, "samples")
    
    def test_add_incompatible_units_raises_error(self):
        """Test that mixing time units and indices raises error"""
        a = Intervals([(0, 100)], units="ms")
        b = Intervals([(200, 300)], units="samples")
        
        with self.assertRaises(ValueError) as cm:
            a + b
        
        self.assertIn("units", str(cm.exception).lower())
    
    def test_add_non_intervals_returns_notimplemented(self):
        """Test that adding non-Intervals returns NotImplemented"""
        a = Intervals([(0, 100)], units="ms")
        
        result = a.__add__("not an Intervals")
        self.assertEqual(result, NotImplemented)
    
    def test_add_empty_left(self):
        """Test adding when left operand is empty"""
        a = Intervals([], units="ms")
        b = Intervals([(200, 300)], units="ms")
        c = a + b
        
        self.assertEqual(len(c), 1)
        self.assertEqual(c.intervals, [(200, 300)])
    
    def test_add_empty_right(self):
        """Test adding when right operand is empty"""
        a = Intervals([(0, 100)], units="ms")
        b = Intervals([], units="ms")
        c = a + b
        
        self.assertEqual(len(c), 1)
        self.assertEqual(c.intervals, [(0, 100)])
    
    def test_add_both_empty(self):
        """Test adding when both operands are empty"""
        a = Intervals([], units="ms")
        b = Intervals([], units="ms")
        c = a + b
        
        self.assertEqual(len(c), 0)
        self.assertEqual(c.units, "ms")
    
    def test_add_label_left_only(self):
        """Test label when only left has label"""
        a = Intervals([(0, 100)], units="ms", label="a")
        b = Intervals([(200, 300)], units="ms")
        c = a + b
        
        self.assertEqual(c.label, "a")
    
    def test_add_label_right_only(self):
        """Test label when only right has label"""
        a = Intervals([(0, 100)], units="ms")
        b = Intervals([(200, 300)], units="ms", label="b")
        c = a + b
        
        self.assertEqual(c.label, "b")
    
    def test_add_chained(self):
        """Test chaining multiple + operations"""
        a = Intervals([(0, 100)], units="ms", label="a")
        b = Intervals([(200, 300)], units="ms", label="b")
        c = Intervals([(400, 500)], units="ms", label="c")
        
        result = a + b + c
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result.intervals, [(0, 100), (200, 300), (400, 500)])
    
class TestGetEpochsAndIntervals(unittest.TestCase):
    """Epochs and Intervals from get_epochs() and get_intervals()"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
    
    def test_return_types(self):
        """get_epochs returns Epochs, get_intervals returns Intervals"""
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms")
        
        self.assertIsInstance(intervals, pp.Epochs)
        self.assertEqual(intervals.window, (-200, 200))
        self.assertIsNotNone(intervals.label)
        # between two events: Intervals
        self.assertIsInstance(self.data.get_intervals(start="F", end="S", units="ms"), Intervals)
    
    def test_get_epochs_has_metadata(self):
        """Test that Epochs from get_epochs have proper metadata"""
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms")
        
        self.assertGreater(len(intervals), 0)
        self.assertEqual(len(intervals.labels), len(intervals))
        self.assertTrue(all("F" in lab for lab in intervals.labels))
    
    def test_get_epochs_different_units(self):
        """plain numbers in the window are in `units`; Epochs are in ms"""
        for units, fac in [("ms", 1), ("sec", 1000), ("min", 60000),
                           ("samples", 1000 / self.data.fs)]:
            intervals = self.data.get_epochs("F", window=(-200, 200), units=units)
            self.assertIsInstance(intervals, pp.Epochs)
            np.testing.assert_allclose(intervals.window, (-200 * fac, 200 * fac))
    
    def test_intervals_label_from_string(self):
        """Test that label is set from string event_select"""
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms")
        self.assertEqual(intervals.label, "F")
    
    def test_intervals_can_be_iterated(self):
        """Test that Epochs from get_epochs can be iterated"""
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms")
        
        count = 0
        for start, end in intervals:
            self.assertIsInstance(start, (int, float, np.number))
            self.assertIsInstance(end, (int, float, np.number))
            self.assertLess(start, end)
            count += 1
        
        self.assertEqual(count, len(intervals))
    
    def test_intervals_custom_label(self):
        """Test that custom label parameter overrides automatic label"""
        custom_label = "my_custom_label"
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms", label=custom_label)
        
        self.assertEqual(intervals.label, custom_label)
    
    def test_intervals_automatic_label_when_none(self):
        """Test that automatic label is used when label parameter is None"""
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms", label=None)
        
        # Should use the automatic label from event_select
        self.assertEqual(intervals.label, "F")


class TestIntervalsPlotMethod(unittest.TestCase):
    """Test Intervals.plot() method"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
        plt.close('all')
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_runs_without_error(self):
        """Test that plot() completes without error"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        
        fig = plt.figure()
        intervals.plot()
        
        # Should have created axes with content
        ax = plt.gca()
        self.assertIsNotNone(ax)
        self.assertGreater(len(ax.get_lines()), 0)
    
    def test_plot_with_labels(self):
        """Test that plot shows labels when requested"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        
        fig = plt.figure()
        intervals.plot(show_labels=True)
        
        ax = plt.gca()
        # Should have text labels
        texts = ax.texts
        self.assertGreater(len(texts), 0)
    
    def test_plot_without_labels(self):
        """Test that plot hides labels when requested"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        
        fig = plt.figure()
        intervals.plot(show_labels=False)
        
        ax = plt.gca()
        # Should not have text labels
        texts = ax.texts
        self.assertEqual(len(texts), 0)
    
    def test_plot_uses_current_axes(self):
        """Test that plot uses the current axes"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # Plot to first axes
        plt.sca(ax1)
        intervals.plot()
        
        # First axes should have lines, second should be empty
        self.assertGreater(len(ax1.get_lines()), 0)
        self.assertEqual(len(ax2.get_lines()), 0)
    
    def test_plot_sets_title_from_label(self):
        """Test that plot uses intervals label as title"""
        intervals = self.data.get_epochs("F", window=(-200, 200), units="ms",
                                            label="test_label").to_intervals()
        
        fig = plt.figure()
        intervals.plot()
        
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "test_label")
    
    def test_plot_sets_xlabel_from_units(self):
        """Test that plot sets x-axis label based on units"""
        # Test with ms
        intervals = spans(self.data, "F", (-200, 200), "ms")
        fig = plt.figure()
        intervals.plot()
        ax = plt.gca()
        self.assertIn("ms", ax.get_xlabel())
        plt.close('all')
        
        # Test with sample indices
        intervals = spans(self.data, "F", (-200, 200), "samples")
        fig = plt.figure()
        intervals.plot()
        ax = plt.gca()
        self.assertIn("Samples", ax.get_xlabel())
    
    def test_plot_handles_empty_intervals(self):
        """Test that plot handles empty intervals gracefully"""
        from pypillometry.intervals import Intervals
        
        empty_intervals = Intervals([], units="ms", label="empty")
        
        fig = plt.figure()
        # Should not raise an error
        empty_intervals.plot()
        
        ax = plt.gca()
        # Should have no lines
        self.assertEqual(len(ax.get_lines()), 0)
    
    def test_plot_with_different_units(self):
        """Test that plot works with different units"""
        for units in ["ms", "sec", "min", "samples"]:
            intervals = spans(self.data, "F", (-200, 200), units)
            
            fig = plt.figure()
            intervals.plot()
            
            ax = plt.gca()
            # Should have created content
            self.assertGreater(len(ax.get_lines()), 0)
            
            plt.close('all')
    
    def test_plot_accepts_color_kwarg(self):
        """Test that interval lines can be colored via kwargs"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        
        fig = plt.figure()
        intervals.plot(color='red')
        
        ax = plt.gca()
        lines = ax.get_lines()
        
        # Check that all lines are red (as specified)
        for line in lines:
            color = line.get_color()
            self.assertEqual(color, 'red')


class TestIntervalsPlotHighlight(unittest.TestCase):
    """Test Intervals.plot_highlight() method"""
    
    def setUp(self):
        """Set up test data"""
        self.intervals = Intervals(
            [(0, 100), (200, 300), (400, 500)], 
            units="ms", 
            label="test"
        )
    
    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')
    
    def test_plot_highlight_single_axes(self):
        """Test plot_highlight on a single axes"""
        fig, ax = plt.subplots()
        ax.plot([0, 600], [0, 1])
        
        patches = self.intervals.plot_highlight()
        
        # Should create 3 patches (one per interval)
        self.assertEqual(len(patches), 3)
    
    def test_plot_highlight_multiple_subplots(self):
        """Test plot_highlight applies to all subplots by default"""
        fig, axs = plt.subplots(3, 1)
        for ax in axs:
            ax.plot([0, 600], [0, 1])
        
        patches = self.intervals.plot_highlight()
        
        # Should create 9 patches (3 intervals x 3 axes)
        self.assertEqual(len(patches), 9)
    
    def test_plot_highlight_specific_axes(self):
        """Test plot_highlight on specific axes only"""
        fig, axs = plt.subplots(2, 1)
        
        patches = self.intervals.plot_highlight(ax=axs[0])
        
        # Should create 3 patches (only on first axes)
        self.assertEqual(len(patches), 3)
    
    def test_plot_highlight_with_figure(self):
        """Test plot_highlight with Figure argument"""
        fig, axs = plt.subplots(2, 1)
        
        patches = self.intervals.plot_highlight(ax=fig)
        
        # Should create 6 patches (3 intervals x 2 axes)
        self.assertEqual(len(patches), 6)

    def test_plot_highlight_zorder(self):
        """zorder defaults to 0 and can be overridden"""
        fig, ax = plt.subplots()
        patches = self.intervals.plot_highlight()
        self.assertTrue(all(p.get_zorder() == 0 for p in patches))
        patches = self.intervals.plot_highlight(zorder=3)
        self.assertTrue(all(p.get_zorder() == 3 for p in patches))
    
    def test_plot_highlight_empty_intervals(self):
        """Test plot_highlight with empty intervals"""
        empty = Intervals([], units="ms")
        fig, ax = plt.subplots()
        
        patches = empty.plot_highlight()
        
        self.assertEqual(len(patches), 0)
    
    def test_plot_highlight_custom_color_alpha(self):
        """Test plot_highlight with custom color and alpha"""
        fig, ax = plt.subplots()
        ax.plot([0, 600], [0, 1])
        
        patches = self.intervals.plot_highlight(color='green', alpha=0.5)
        
        self.assertEqual(len(patches), 3)
        # Check that patches have correct color
        for patch in patches:
            fc = patch.get_facecolor()
            # Green should have high G component
            self.assertGreater(fc[1], 0.4)  # G component
    
    def test_plot_highlight_returns_patches(self):
        """Test that plot_highlight returns matplotlib patches"""
        from matplotlib.patches import Patch
        
        fig, ax = plt.subplots()
        patches = self.intervals.plot_highlight()
        
        # All returned items should be patches
        for patch in patches:
            self.assertIsInstance(patch, Patch)
    
    def test_plot_highlight_with_list_of_axes(self):
        """Test plot_highlight with explicit list of axes"""
        fig, axs = plt.subplots(3, 1)
        
        # Only highlight first two
        patches = self.intervals.plot_highlight(ax=[axs[0], axs[1]])
        
        # Should create 6 patches (3 intervals x 2 axes)
        self.assertEqual(len(patches), 6)


class TestIntervalsConversionMethods(unittest.TestCase):
    """Test Intervals conversion methods (as_index, to_units, __array__)"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
    
    def test_array_conversion_with_units_none(self):
        """Test __array__() with index units"""
        intervals = spans(self.data, "F", (-200, 200), "samples")
        arr = np.array(intervals)
        
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape[1], 2)  # Should have 2 columns (start, end)
        self.assertEqual(len(arr), len(intervals))
    
    def test_array_conversion_with_units_ms(self):
        """Test __array__() with time units"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        arr = np.array(intervals)
        
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape[1], 2)
        self.assertEqual(len(arr), len(intervals))
    
    def test_to_indices_with_units_none(self):
        """Test to_units('indices') when intervals already in indices"""
        intervals = spans(self.data, "F", (-200, 200), "samples")
        indices_intervals = intervals.to_units("indices")
        indices = np.array(indices_intervals).astype(int)
        
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.dtype, np.int_)
        self.assertEqual(indices.shape[1], 2)
        self.assertEqual(len(indices), len(intervals))
    
    def test_to_indices_with_units_ms(self):
        """Test to_units('indices') with millisecond units"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        indices_intervals = intervals.to_units("indices", self.data)
        indices = np.array(indices_intervals).astype(int)
        
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.dtype, np.int_)
        self.assertEqual(indices.shape[1], 2)
        
        # Check that indices are valid for the data
        for start, end in indices:
            self.assertGreaterEqual(start, 0)
            self.assertLess(end, len(self.data.tx))

    def test_to_indices_with_units_sec(self):
        """Test to_units('indices') with second units"""
        intervals = spans(self.data, "F", (-0.2, 0.2), "sec")
        indices = np.array(intervals.to_units("indices", self.data)).astype(int)
        np.testing.assert_array_equal(
            indices, np.array(spans(self.data, "F", (-0.2, 0.2), "sec")
                              .to_units("ms").to_units("indices", self.data)))
        for start, end in indices:
            self.assertGreaterEqual(start, 0)
            self.assertLess(end, len(self.data.tx))

    def test_to_units_ms_to_sec(self):
        """Test converting from milliseconds to seconds"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        intervals_sec = intervals.to_units("sec")
        
        self.assertEqual(intervals_sec.units, "sec")
        self.assertEqual(len(intervals_sec), len(intervals))
        
        # Check conversion accuracy (200ms = 0.2sec)
        arr_ms = np.array(intervals)
        arr_sec = np.array(intervals_sec)
        np.testing.assert_array_almost_equal(arr_sec, arr_ms / 1000.0, decimal=6)
    
    def test_to_units_sec_to_ms(self):
        """Test converting from seconds to milliseconds"""
        intervals = spans(self.data, "F", (-0.2, 0.2), "sec")
        intervals_ms = intervals.to_units("ms")
        
        self.assertEqual(intervals_ms.units, "ms")
        
        # Check conversion accuracy
        arr_sec = np.array(intervals)
        arr_ms = np.array(intervals_ms)
        np.testing.assert_array_almost_equal(arr_ms, arr_sec * 1000.0, decimal=6)
    
    def test_to_units_ms_to_min(self):
        """Test converting from milliseconds to minutes"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        intervals_min = intervals.to_units("min")
        
        self.assertEqual(intervals_min.units, "min")
        
        # Check conversion accuracy (200ms = 0.2/60 min)
        arr_ms = np.array(intervals)
        arr_min = np.array(intervals_min)
        np.testing.assert_array_almost_equal(arr_min, arr_ms / 60000.0, decimal=9)
    
    def test_to_units_same_units(self):
        """Test converting to same units returns self"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        intervals_same = intervals.to_units("ms")
        
        # Should return the same object
        self.assertIs(intervals_same, intervals)
    
    def test_to_units_with_aliases(self):
        """Test converting to units using aliases"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        
        # Test various aliases
        intervals_seconds = intervals.to_units("seconds")
        intervals_s = intervals.to_units("s")
        intervals_minutes = intervals.to_units("minutes")
        intervals_hrs = intervals.to_units("hrs")
        
        # Check that canonical units are returned
        self.assertEqual(intervals_seconds.units, "sec")
        self.assertEqual(intervals_s.units, "sec")
        self.assertEqual(intervals_minutes.units, "min")
        self.assertEqual(intervals_hrs.units, "h")
        
        # Check conversion accuracy
        arr_ms = np.array(intervals)
        arr_seconds = np.array(intervals_seconds)
        arr_s = np.array(intervals_s)
        
        np.testing.assert_array_almost_equal(arr_seconds, arr_ms / 1000.0, decimal=6)
        np.testing.assert_array_almost_equal(arr_s, arr_ms / 1000.0, decimal=6)
    
    def test_to_units_from_indices_uses_time_vector(self):
        """converting from indices uses the time vector of the data object"""
        intervals = spans(self.data, "F", (-200, 200), "samples")
        intervals_ms = intervals.to_units("ms", self.data)
        self.assertEqual(intervals_ms.units, "ms")
        start, end = intervals.intervals[0]
        self.assertEqual(intervals_ms.intervals[0], (self.data.tx[start], self.data.tx[end]))
        # the time of the event sample
        self.assertEqual(intervals_ms.event_onsets[0], self.data.tx[intervals.event_onsets[0]])

    def test_to_units_from_indices_raises_error_without_data(self):
        """converting from indices needs the data object"""
        intervals = Intervals([(0, 100), (200, 300)], units="samples")
        with self.assertRaises(ValueError) as cm:
            intervals.to_units("ms")
        self.assertIn("data object", str(cm.exception))

    def test_to_units_to_indices_uses_time_vector(self):
        """a time is converted to the first sample at or after it"""
        d = timebase(10, fs=100., t0=1000.)  # samples at 1000, 1010, ..., 1090 ms
        intervals = Intervals([(1000, 1020), (1015, 1100)], units="ms",
                              event_onsets=[1005, 1015])
        idx = intervals.to_units("indices", d)
        self.assertEqual(idx.units, "samples")
        self.assertEqual(idx.intervals, [(0, 2), (2, 10)])
        self.assertEqual(list(idx.event_onsets), [1, 2])
        # and back: each index is the time of its sample
        back = idx.to_units("ms", d)
        self.assertEqual(back.intervals, [(1000., 1020.), (1020., 1100.)])

    def test_to_units_to_indices_raises_error_without_data(self):
        """converting to indices needs the data object"""
        intervals = Intervals([(0, 100), (200, 300)], units="ms")
        with self.assertRaises(ValueError) as cm:
            intervals.to_units("indices")
        self.assertIn("data object", str(cm.exception))

    def test_to_units_unknown_source_raises_error(self):
        """Test that unknown source units raise error in constructor"""
        # Now that units are normalized in __init__, invalid units are caught there
        with self.assertRaises(ValueError) as cm:
            intervals = Intervals([(0, 100), (200, 300)], units="unknown", label="test")
        
        self.assertIn("unknown", str(cm.exception).lower())
    
    def test_to_units_unknown_target_raises_error(self):
        """Test that unknown target units raise error"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        
        with self.assertRaises(ValueError) as cm:
            intervals.to_units("unknown")
        
        self.assertIn("unknown", str(cm.exception).lower())
    
    def test_to_units_preserves_metadata(self):
        """Test that to_units preserves metadata"""
        intervals = spans(self.data, "F", (-200, 200), "ms")
        intervals_sec = intervals.to_units("sec")
        
        # Check that metadata is preserved
        self.assertEqual(intervals_sec.label, intervals.label)
        if intervals.event_labels is not None:
            self.assertEqual(len(intervals_sec.event_labels), len(intervals.event_labels))


class TestConcatIntervals(unittest.TestCase):
    """Test concat_intervals function"""
    
    def test_concat_intervals_from_args(self):
        """Test concatenating multiple Intervals passed as arguments"""
        intervals1 = Intervals([(0, 100), (200, 300)], units="ms", label="int1",
                              event_labels=["a", "b"])
        intervals2 = Intervals([(50, 150), (400, 500)], units="ms", label="int2",
                              event_labels=["c", "d"])
        
        merged = concat_intervals(intervals1, intervals2)
        
        self.assertIsInstance(merged, Intervals)
        self.assertEqual(len(merged), 4)  # All 4 intervals, not merged
        self.assertEqual(merged.units, "ms")
        self.assertEqual(merged.label, "combined")
        
        # Check metadata preserved
        self.assertIsNotNone(merged.event_labels)
        self.assertEqual(merged.event_labels, ["a", "b", "c", "d"])
    
    def test_concat_intervals_from_list(self):
        """Test concatenating from a list of Intervals"""
        intervals1 = Intervals([(0, 100)], units="ms", event_labels=["x"])
        intervals2 = Intervals([(200, 300)], units="ms", event_labels=["y"])
        intervals3 = Intervals([(400, 500)], units="ms", event_labels=["z"])
        
        intervals_list = [intervals1, intervals2, intervals3]
        merged = concat_intervals(intervals_list)
        
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged.event_labels, ["x", "y", "z"])
    
    def test_concat_intervals_from_dict(self):
        """Test concatenating from a dict of Intervals"""
        intervals1 = Intervals([(0, 100)], units="ms", event_labels=["p"])
        intervals2 = Intervals([(200, 300)], units="ms", event_labels=["q"])
        
        intervals_dict = {"left_pupil": intervals1, "right_pupil": intervals2}
        merged = concat_intervals(intervals_dict)
        
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.event_labels, ["p", "q"])
    
    def test_concat_intervals_custom_label(self):
        """Test concatenating with custom label"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(200, 300)], units="ms")
        
        merged = concat_intervals(intervals1, intervals2, label="custom")
        
        self.assertEqual(merged.label, "custom")
    
    def test_concat_intervals_preserves_all_metadata(self):
        """Test that all metadata is preserved"""
        intervals1 = Intervals([(0, 100), (200, 300)], units="ms",
                              event_labels=["a", "b"],
                              event_onsets=[10, 20])
        intervals2 = Intervals([(400, 500)], units="ms",
                              event_labels=["c"],
                              event_onsets=[30])
        
        merged = concat_intervals(intervals1, intervals2)
        
        self.assertEqual(merged.event_labels, ["a", "b", "c"])
        self.assertEqual(list(merged.event_onsets), [10, 20, 30])
    
    def test_concat_intervals_partial_metadata(self):
        """Test concatenating when only some objects have metadata"""
        intervals1 = Intervals([(0, 100)], units="ms",
                              event_labels=["a"])
        intervals2 = Intervals([(200, 300)], units="ms")  # No metadata
        intervals3 = Intervals([(400, 500)], units="ms",
                              event_labels=["c"])
        
        merged = concat_intervals(intervals1, intervals2, intervals3)
        
        # Should preserve labels with "" for missing
        self.assertEqual(merged.event_labels, ["a", "", "c"])
    
    def test_concat_intervals_no_metadata(self):
        """Test concatenating when no objects have metadata"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(200, 300)], units="ms")
        
        merged = concat_intervals(intervals1, intervals2)
        
        # Should have no metadata
        self.assertIsNone(merged.event_labels)
        self.assertIsNone(merged.event_onsets)
    
    def test_concat_intervals_does_not_merge_overlaps(self):
        """Test that overlapping intervals are NOT merged, just concatenated"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(50, 150)], units="ms")  # Overlaps with first
        
        merged = concat_intervals(intervals1, intervals2)
        
        # Should have both intervals, not merged
        self.assertEqual(len(merged), 2)
        arr = np.array(merged)
        self.assertEqual(arr[0, 0], 0)
        self.assertEqual(arr[0, 1], 100)
        self.assertEqual(arr[1, 0], 50)
        self.assertEqual(arr[1, 1], 150)
    
    def test_concat_intervals_then_merge_method(self):
        """Test combining intervals then merging overlaps with .merge()"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(50, 150)], units="ms")
        
        combined = concat_intervals(intervals1, intervals2)
        self.assertEqual(len(combined), 2)
        
        # Now merge overlaps
        merged = combined.merge()
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0], (0, 150))
    
    def test_concat_intervals_empty_intervals(self):
        """Test concatenating when one or more Intervals are empty"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([], units="ms")
        intervals3 = Intervals([(200, 300)], units="ms")
        
        merged = concat_intervals(intervals1, intervals2, intervals3)
        
        self.assertEqual(len(merged), 2)
    
    def test_concat_intervals_all_empty(self):
        """Test concatenating when all Intervals are empty"""
        intervals1 = Intervals([], units="ms")
        intervals2 = Intervals([], units="ms")
        
        merged = concat_intervals(intervals1, intervals2)
        
        self.assertEqual(len(merged), 0)
        self.assertEqual(merged.units, "ms")
    
    def test_concat_intervals_converts_units(self):
        """Time units are converted to those of the first object, as with +"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(0, 1)], units="sec")
        combined = concat_intervals(intervals1, intervals2)
        self.assertEqual(combined.units, "ms")
        self.assertEqual(combined.intervals, [(0, 100), (0.0, 1000.0)])
        # time units and sample indices cannot be combined
        with self.assertRaisesRegex(ValueError, "units"):
            concat_intervals(intervals1, Intervals([(0, 10)], units="samples"))

    def test_concat_intervals_same_as_add(self):
        """concat_intervals(a, b, c) gives the intervals and metadata of a + b + c"""
        a = Intervals([(0, 100)], units="ms", event_labels=["a"], event_meta={"x": [1]})
        b = Intervals([(200, 300)], units="ms")
        c = Intervals([(1, 2)], units="sec", event_labels=["c"], event_onsets=[1])
        added, combined = a + b + c, concat_intervals([a, b, c])
        self.assertEqual(combined.intervals, added.intervals)
        self.assertEqual(combined.event_labels, added.event_labels)
        self.assertEqual(list(combined.event_onsets), list(added.event_onsets))
        np.testing.assert_array_equal(combined.event_meta["x"], added.event_meta["x"])
    
    def test_concat_intervals_single_object(self):
        """Test concatenating with a single Intervals object"""
        intervals = Intervals([(0, 100), (200, 300)], units="ms",
                             event_labels=["a", "b"])
        
        merged = concat_intervals(intervals)
        
        # Should return a new object with same content
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.event_labels, ["a", "b"])
    
    def test_concat_intervals_no_arguments_raises_error(self):
        """Test that calling with no arguments raises error"""
        with self.assertRaises(ValueError) as cm:
            concat_intervals()
        
        self.assertIn("no intervals", str(cm.exception).lower())
    
    def test_concat_intervals_invalid_type_raises_error(self):
        """Test that invalid argument types raise error"""
        intervals1 = Intervals([(0, 100)], units="ms")
        
        with self.assertRaises(TypeError):
            concat_intervals(intervals1, "not an intervals object")
    
    def test_concat_intervals_different_units(self):
        """Test concatenating with different unit types"""
        for units in ["ms", "sec", "min", "h", "samples"]:
            intervals1 = Intervals([(0, 100)], units=units)
            intervals2 = Intervals([(200, 300)], units=units)
            
            merged = concat_intervals(intervals1, intervals2)
            
            self.assertEqual(merged.units, units)
            self.assertEqual(len(merged), 2)


class TestIntervalsMaskConversion(unittest.TestCase):
    """Test Intervals to_mask() and from_mask() methods"""
    
    def test_to_mask_basic(self):
        """Test basic conversion to mask"""
        intervals = Intervals([(0, 10), (20, 30)], units="samples")
        mask = intervals.to_mask(timebase(50))
        
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(len(mask), 50)
        self.assertEqual(mask.dtype, np.int8)
        
        # Check mask values
        np.testing.assert_array_equal(mask[0:10], 1)
        np.testing.assert_array_equal(mask[10:20], 0)
        np.testing.assert_array_equal(mask[20:30], 1)
        np.testing.assert_array_equal(mask[30:50], 0)



    def test_to_mask_empty_intervals(self):
        """Test to_mask with empty intervals"""
        mask = Intervals([], units="samples").to_mask(timebase(100))
        self.assertEqual(len(mask), 100)
        np.testing.assert_array_equal(mask, 0)

    def test_to_mask_converts_time_units(self):
        """to_mask converts time-based intervals with the time vector of the data"""
        intervals = Intervals([(0, 100), (200, 300)], units="ms")
        mask = intervals.to_mask(timebase(500))
        self.assertEqual(len(mask), 500)
        np.testing.assert_array_equal(mask[0:100], 1)
        np.testing.assert_array_equal(mask[100:200], 0)
        np.testing.assert_array_equal(mask[200:300], 1)
        np.testing.assert_array_equal(mask[300:500], 0)
        # recording that does not start at 0
        mask = intervals.to_mask(timebase(500, t0=150.))
        np.testing.assert_array_equal(np.where(mask)[0], np.arange(50, 150))



    def test_to_mask_needs_data(self):
        """to_mask needs the data object"""
        with self.assertRaises(TypeError):
            Intervals([(0, 10)], units="samples").to_mask()

    def test_to_mask_clips_to_bounds(self):
        """Test that intervals outside mask bounds are clipped"""
        intervals = Intervals([(-10, 10), (40, 60)], units="samples")
        mask = intervals.to_mask(timebase(50))
        np.testing.assert_array_equal(np.where(mask)[0], list(range(10)) + list(range(40, 50)))

    def test_from_mask_basic(self):
        """Test basic creation from mask"""
        mask = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        intervals = Intervals.from_mask(mask)
        
        self.assertIsInstance(intervals, Intervals)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals.intervals, [(2, 5), (7, 9)])
        self.assertEqual(intervals.units, "samples")
    
    def test_from_mask_all_zeros(self):
        """Test from_mask with all zeros"""
        mask = np.zeros(100, dtype=int)
        intervals = Intervals.from_mask(mask)
        
        self.assertEqual(len(intervals), 0)
    
    def test_from_mask_all_ones(self):
        """Test from_mask with all ones"""
        mask = np.ones(100, dtype=int)
        intervals = Intervals.from_mask(mask)
        
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals.intervals, [(0, 100)])
    
    def test_from_mask_with_label(self):
        """Test from_mask with label"""
        mask = np.array([1, 1, 0, 0, 1, 1])
        intervals = Intervals.from_mask(mask, label="test_mask")
        
        self.assertEqual(intervals.label, "test_mask")
    
    def test_from_mask_checks_length(self):
        """the mask must have one entry per sample of the data"""
        with self.assertRaises(ValueError):
            Intervals.from_mask(np.array([1, 1, 0]), timebase(4))

    def test_from_mask_time_units(self):
        """from_mask with time units converts with the time vector of the data"""
        mask = np.array([0, 1, 1, 0])
        intervals = Intervals.from_mask(mask, timebase(4, fs=500.), units="ms")
        self.assertEqual(intervals.units, "ms")
        self.assertEqual(intervals.intervals, [(2.0, 6.0)])
        intervals = Intervals.from_mask(mask, timebase(4, fs=500., t0=100.), units="ms")
        self.assertEqual(intervals.intervals, [(102.0, 106.0)])

    def test_from_mask_time_units_without_data(self):
        """from_mask with time units but no data object raises"""
        with self.assertRaises(ValueError):
            Intervals.from_mask(np.array([0, 1, 1, 0]), units="ms")

    def test_from_mask_boolean_input(self):
        """Test from_mask with boolean array"""
        mask = np.array([False, True, True, False, True])
        intervals = Intervals.from_mask(mask)
        
        self.assertEqual(intervals.intervals, [(1, 3), (4, 5)])
    
    def test_roundtrip_to_mask_from_mask(self):
        """Test roundtrip: intervals -> mask -> intervals"""
        original = Intervals([(10, 20), (30, 50), (70, 80)], units="samples")
        recovered = Intervals.from_mask(original.to_mask(timebase(100)))
        self.assertEqual(original.intervals, recovered.intervals)

class TestIntervalsFilter(unittest.TestCase):
    """Test Intervals.filter() method"""
    
    def test_filter_by_min_duration(self):
        """Test filtering by minimum duration"""
        intervals = Intervals([(0, 50), (100, 200), (300, 500)], units="ms")
        
        filtered = intervals.filter(min_duration=100)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.intervals, [(100, 200), (300, 500)])
    
    def test_filter_by_max_duration(self):
        """Test filtering by maximum duration"""
        intervals = Intervals([(0, 50), (100, 200), (300, 500)], units="ms")
        
        filtered = intervals.filter(max_duration=100)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.intervals, [(0, 50), (100, 200)])
    
    def test_filter_by_duration_range(self):
        """Test filtering by both min and max duration"""
        intervals = Intervals([(0, 50), (100, 200), (300, 500), (600, 610)], units="ms")
        
        filtered = intervals.filter(min_duration=50, max_duration=150)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.intervals, [(0, 50), (100, 200)])
    
    def test_filter_by_time_range_overlap(self):
        """Test filtering by time range (overlap)"""
        intervals = Intervals([(0, 100), (200, 300), (400, 500)], units="ms")
        
        # Should keep intervals that overlap with (150, 350)
        filtered = intervals.filter(time_range=(150, 350))
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.intervals, [(200, 300)])
    
    def test_filter_by_time_range_partial_overlap(self):
        """Test that partial overlap is sufficient"""
        intervals = Intervals([(0, 200), (300, 400)], units="ms")
        
        # (0, 200) partially overlaps (150, 250)
        filtered = intervals.filter(time_range=(150, 250))
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.intervals, [(0, 200)])
    
    def test_filter_by_time_range_no_overlap(self):
        """Test filtering when no intervals overlap"""
        intervals = Intervals([(0, 100), (400, 500)], units="ms")
        
        filtered = intervals.filter(time_range=(150, 350))
        
        self.assertEqual(len(filtered), 0)
    
    def test_filter_by_selector_function(self):
        """Test filtering with custom selector function"""
        intervals = Intervals([(0, 100), (100, 200), (200, 300)], units="ms")
        
        # Keep intervals starting at multiples of 200
        filtered = intervals.filter(selector=lambda s, e: s % 200 == 0)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.intervals, [(0, 100), (200, 300)])
    
    def test_filter_by_selector_using_end(self):
        """Test filtering with selector using end value"""
        intervals = Intervals([(0, 100), (100, 250), (200, 300)], units="ms")
        
        # Keep intervals ending at multiples of 100
        filtered = intervals.filter(selector=lambda s, e: e % 100 == 0)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.intervals, [(0, 100), (200, 300)])
    
    def test_filter_combined_criteria(self):
        """Test filtering with multiple criteria combined"""
        intervals = Intervals([(0, 50), (100, 200), (300, 500), (600, 700)], units="ms")
        
        # min_duration=50, max_duration=150, time_range=(50, 400)
        filtered = intervals.filter(
            min_duration=50, 
            max_duration=150, 
            time_range=(50, 400)
        )
        
        # (0, 50): duration=50 OK, but doesn't overlap (50, 400) - NO
        # (100, 200): duration=100 OK, overlaps (50, 400) - YES
        # (300, 500): duration=200 > 150 - NO
        # (600, 700): duration=100 OK, but doesn't overlap (50, 400) - NO
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.intervals, [(100, 200)])
    
    def test_filter_preserves_metadata(self):
        """Test that filter preserves all metadata"""
        intervals = Intervals(
            [(0, 100), (200, 300), (400, 500)], 
            units="ms", 
            label="test_label",
            event_labels=["a", "b", "c"],
            event_onsets=np.array([50, 250, 450]))
        
        filtered = intervals.filter(min_duration=100)
        
        self.assertEqual(filtered.units, "ms")
        self.assertEqual(filtered.label, "test_label")
        self.assertEqual(filtered.event_labels, ["a", "b", "c"])
        np.testing.assert_array_equal(filtered.event_onsets, [50, 250, 450])
    
    def test_filter_filters_metadata_correctly(self):
        """Test that metadata is filtered in sync with intervals"""
        intervals = Intervals(
            [(0, 50), (200, 300), (400, 500)], 
            units="ms",
            event_labels=["short", "medium", "long"],
            event_onsets=np.array([25, 250, 450])
        )
        
        # Filter out the short one
        filtered = intervals.filter(min_duration=100)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.event_labels, ["medium", "long"])
        np.testing.assert_array_equal(filtered.event_onsets, [250, 450])
    
    def test_filter_empty_intervals(self):
        """Test filtering empty intervals"""
        intervals = Intervals([], units="ms")
        
        filtered = intervals.filter(min_duration=100)
        
        self.assertEqual(len(filtered), 0)
        self.assertEqual(filtered.units, "ms")
    
    def test_filter_no_criteria_returns_all(self):
        """Test that filter with no criteria returns all intervals"""
        intervals = Intervals([(0, 100), (200, 300)], units="ms")
        
        filtered = intervals.filter()
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.intervals, [(0, 100), (200, 300)])
    
    def test_filter_all_filtered_out(self):
        """Test filtering that removes all intervals"""
        intervals = Intervals([(0, 10), (20, 30)], units="ms")
        
        filtered = intervals.filter(min_duration=100)
        
        self.assertEqual(len(filtered), 0)
    
    def test_filter_time_range_invalid_raises_error(self):
        """Test that invalid time_range raises error"""
        intervals = Intervals([(0, 100)], units="ms")
        
        with self.assertRaises(ValueError) as cm:
            intervals.filter(time_range=(100, 50))  # start >= end
        
        self.assertIn("start", str(cm.exception).lower())
    
    def test_filter_time_range_wrong_length_raises_error(self):
        """Test that time_range with wrong length raises error"""
        intervals = Intervals([(0, 100)], units="ms")
        
        with self.assertRaises(ValueError) as cm:
            intervals.filter(time_range=(100,))
        
        self.assertIn("time_range", str(cm.exception).lower())
    
    def test_filter_non_callable_selector_raises_error(self):
        """Test that non-callable selector raises error"""
        intervals = Intervals([(0, 100)], units="ms")
        
        with self.assertRaises(TypeError) as cm:
            intervals.filter(selector="not a function")
        
        self.assertIn("callable", str(cm.exception).lower())
    
    def test_filter_with_indices_units(self):
        """Test filtering with index-based intervals"""
        intervals = Intervals([(0, 50), (100, 200), (300, 500)], units="samples")
        
        filtered = intervals.filter(min_duration=100)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.units, "samples")
    
    def test_filter_returns_new_object(self):
        """Test that filter returns a new Intervals object"""
        intervals = Intervals([(0, 100), (200, 300)], units="ms")
        
        filtered = intervals.filter()
        
        self.assertIsNot(filtered, intervals)
        self.assertIsNot(filtered.intervals, intervals.intervals)
    
    def test_filter_duration_boundary_inclusive(self):
        """Test that duration boundaries are inclusive"""
        intervals = Intervals([(0, 100), (200, 300)], units="ms")
        
        # Exactly 100ms duration
        filtered = intervals.filter(min_duration=100, max_duration=100)
        
        self.assertEqual(len(filtered), 2)
    
    def test_filter_time_range_boundary_exclusive(self):
        """Test time range boundary behavior"""
        intervals = Intervals([(0, 100), (100, 200), (200, 300)], units="ms")
        
        # Range (100, 200) - start exclusive, end exclusive for overlap
        # Overlap means: interval.end > range.start AND interval.start < range.end
        filtered = intervals.filter(time_range=(100, 200))
        
        # (0, 100): end=100 > 100? No - excluded
        # (100, 200): end=200 > 100? Yes, start=100 < 200? Yes - included
        # (200, 300): start=200 < 200? No - excluded
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.intervals, [(100, 200)])


class TestIntervalsComplement(unittest.TestCase):
    """Intervals.complement()"""

    def test_samples_with_data(self):
        iv = Intervals([(10, 20), (50, 60)], units="samples")
        c = iv.complement(timebase(100))
        self.assertEqual(c.intervals, [(0, 10), (20, 50), (60, 100)])

    def test_masks_add_up(self):
        d = timebase(100, fs=100., t0=500.)
        iv = Intervals([(600, 700), (900, 1000)], units="ms")
        c = iv.complement(d)
        np.testing.assert_array_equal(iv.to_mask(d) + c.to_mask(d), 1)

    def test_time_units_cover_last_sample(self):
        d = timebase(100, fs=100.)  # samples at 0, 10, ..., 990 ms
        c = Intervals([(0, 500)], units="ms").complement(d)
        self.assertEqual(c.intervals, [(500.0, 1000.0)])
        self.assertEqual(c.to_units("sec").intervals, [(0.5, 1.0)])
        c = Intervals([(0, 0.5)], units="sec").complement(d)
        self.assertEqual(c.intervals, [(0.5, 1.0)])

    def test_overlapping_and_outside(self):
        iv = Intervals([(-50, 10), (5, 30), (80, 200)], units="samples")
        self.assertEqual(iv.complement(timebase(100)).intervals, [(30, 80)])

    def test_empty(self):
        c = Intervals([], units="samples").complement(timebase(100))
        self.assertEqual(c.intervals, [(0, 100)])

    def test_everything_covered(self):
        c = Intervals([(0, 100)], units="samples").complement(timebase(100))
        self.assertEqual(c.intervals, [])

    def test_label(self):
        iv = Intervals([(10, 20)], units="samples", label="blocks")
        self.assertEqual(iv.complement(timebase(100)).label, "not blocks")

    def test_needs_data(self):
        with self.assertRaises(TypeError):
            Intervals([(10, 20)], units="ms").complement()


class TestIntervalsSubtractOperator(unittest.TestCase):
    """Test Intervals - operator for subtracting intervals"""
    
    def test_subtract_basic(self):
        """Test basic subtraction of intervals"""
        a = Intervals([(0, 100), (200, 300)], units="samples")
        b = Intervals([(50, 150)], units="samples")
        c = a - b
        
        self.assertIsInstance(c, Intervals)
        # (0, 100) - (50, 150) = (0, 50)
        # (200, 300) unchanged
        self.assertEqual(c.intervals, [(0, 50), (200, 300)])
    
    def test_subtract_complete_removal(self):
        """Test subtraction that completely removes an interval"""
        a = Intervals([(50, 100)], units="samples")
        b = Intervals([(0, 150)], units="samples")
        c = a - b
        
        # (50, 100) is completely contained in (0, 150), so nothing remains
        self.assertEqual(len(c), 0)
    
    def test_subtract_no_overlap(self):
        """Test subtraction with no overlap"""
        a = Intervals([(0, 50), (100, 150)], units="samples")
        b = Intervals([(60, 90)], units="samples")
        c = a - b
        
        # No overlap, a unchanged
        self.assertEqual(c.intervals, [(0, 50), (100, 150)])
    
    def test_subtract_splits_interval(self):
        """Test subtraction that splits an interval in two"""
        a = Intervals([(0, 100)], units="samples")
        b = Intervals([(40, 60)], units="samples")
        c = a - b
        
        # (0, 100) - (40, 60) = (0, 40) and (60, 100)
        self.assertEqual(c.intervals, [(0, 40), (60, 100)])
    
    def test_subtract_empty_left(self):
        """Test subtraction with empty left operand"""
        a = Intervals([], units="samples")
        b = Intervals([(20, 50)], units="samples")
        c = a - b
        
        self.assertEqual(len(c), 0)
    
    def test_subtract_empty_right(self):
        """Test subtraction with empty right operand"""
        a = Intervals([(0, 50)], units="samples")
        b = Intervals([], units="samples")
        c = a - b
        
        self.assertEqual(c.intervals, [(0, 50)])
    
    def test_subtract_with_time_units(self):
        """Test subtraction with time-based intervals"""
        # 1000Hz sampling rate
        a = Intervals([(0, 100), (200, 300)], units="ms")
        b = Intervals([(50, 150)], units="ms")
        c = a - b
        
        # Result should also be in ms (converted back from indices)
        self.assertEqual(c.units, "ms")
        # Check intervals are approximately correct
        self.assertEqual(len(c), 2)
        self.assertAlmostEqual(c.intervals[0][0], 0, places=0)
        self.assertAlmostEqual(c.intervals[0][1], 50, places=0)
        self.assertAlmostEqual(c.intervals[1][0], 200, places=0)
        self.assertAlmostEqual(c.intervals[1][1], 300, places=0)
    
    def test_subtract_creates_label(self):
        """Test that subtraction creates combined label"""
        a = Intervals([(0, 100)], units="samples", label="a")
        b = Intervals([(50, 75)], units="samples", label="b")
        c = a - b
        
        self.assertIn("a", c.label)
        self.assertIn("b", c.label)
    
    def test_subtract_mixed_units(self):
        """time units are converted; indices and time units cannot be combined"""
        a = Intervals([(0, 1)], units="sec")
        b = Intervals([(500, 1500)], units="ms")
        self.assertEqual((a - b).intervals, [(0, 0.5)])
        with self.assertRaises(ValueError):
            a - Intervals([(0, 10)], units="samples")

    def test_subtract_non_intervals_returns_notimplemented(self):
        """Test that subtracting non-Intervals returns NotImplemented"""
        a = Intervals([(0, 100)], units="samples")
        
        result = a.__sub__("not an Intervals")
        self.assertEqual(result, NotImplemented)
    
    def test_subtract_overlapping_left(self):
        """overlapping intervals on the left are merged first"""
        a = Intervals([(0, 100), (90, 120)], units="samples")
        b = Intervals([(50, 75)], units="samples")
        self.assertEqual((a - b).intervals, [(0, 50), (75, 120)])

    def test_subtract_multiple_from_multiple(self):
        """Test subtracting multiple intervals from multiple intervals"""
        a = Intervals([(0, 50), (100, 150), (200, 250)], units="samples")
        b = Intervals([(25, 75), (125, 225)], units="samples")
        c = a - b
        
        # (0, 50) - (25, 75) = (0, 25)
        # (100, 150) - (125, 225) = (100, 125)
        # (200, 250) - (125, 225) = (225, 250)
        self.assertEqual(c.intervals, [(0, 25), (100, 125), (225, 250)])


if __name__ == '__main__':
    unittest.main()



class TestConversionWithData(unittest.TestCase):
    """Conversions use the time vector of the data (recording not starting at 0)."""

    def setUp(self):
        self.d = pp.EyeData(time=1000 + np.arange(0, 10000, 10.), left_pupil=np.ones(1000),
                              event_onsets=[2000, 5005], event_labels=["a", "b"])

    def test_get_epochs_samples_and_ms_agree(self):
        ms = self.d.get_epochs("", (-100, 200), units="ms")
        ix = self.d.get_epochs("", (-10, 20), units="samples")
        self.assertEqual(ix, ms)
        self.assertEqual(ms.to_intervals("samples", self.d).intervals,
                         [(90, 120), (391, 421)])

    def test_mask_intervals_in_ms(self):
        d = self.d.mask_intervals(Intervals([(1100, 1150)], units="ms"), inplace=False)
        np.testing.assert_array_equal(np.where(d.data.mask["left_pupil"])[0], np.arange(10, 15))

    def test_segments_from_samples_and_ms(self):
        seg_ms = self.d.get_segments(self.d.get_epochs("", (-100, 200), units="ms"), "left_pupil")
        seg_ix = self.d.get_segments(self.d.get_epochs("", (-10, 20), units="samples"), "left_pupil")
        np.testing.assert_allclose(seg_ms.tx, seg_ix.tx)
        self.assertEqual(seg_ms.data.shape, seg_ix.data.shape)

    def test_units_none_rejected(self):
        """sample indices are units="samples"; units=None raises"""
        with self.assertRaisesRegex(ValueError, "samples"):
            Intervals([(0, 10)], units=None)
        with self.assertRaisesRegex(ValueError, "samples"):
            self.d.get_epochs("", (-10, 20), units=None)
        with self.assertRaisesRegex(ValueError, "samples"):
            self.d.get_blinks("left", "pupil", units=None)["left_pupil"]
        with self.assertRaisesRegex(ValueError, "samples"):
            Intervals([(0, 10)], units="ms").to_units(None)
        self.assertEqual(Intervals([(0, 10)], units="indices").units, "samples")

    def test_samples_helper(self):
        """durations -> number of samples (100 Hz)"""
        self.assertEqual(self.d._samples(50), 5)
        self.assertEqual(self.d._samples(55), 5)
        self.assertEqual(self.d._samples(55, rounding="up"), 6)
        self.assertEqual(self.d._samples(0.07, "sec"), 7)  # no rounding error
        self.assertEqual(self.d._samples(70 * pp.u.ms, rounding="up"), 7)
        self.assertEqual(self.d._samples((20, "0.1 sec")), (2, 10))
