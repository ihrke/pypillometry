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
from pypillometry.intervals import Intervals, IntervalStats, merge_intervals


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
        
        arr = intervals.to_array()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (2, 2))
        np.testing.assert_array_equal(arr, [[0, 100], [200, 300]])
    
    def test_intervals_with_metadata(self):
        """Test Intervals with event metadata"""
        intervals_list = [(0, 100), (200, 300)]
        event_labels = ["event1", "event2"]
        event_indices = np.array([0, 5])
        
        intervals = Intervals(
            intervals_list, 
            units="ms", 
            label="stimuli",
            event_labels=event_labels,
            event_indices=event_indices
        )
        
        self.assertEqual(intervals.label, "stimuli")
        self.assertEqual(intervals.event_labels, ["event1", "event2"])
        np.testing.assert_array_equal(intervals.event_indices, [0, 5])

    def test_intervals_as_pandas(self):
        """Test conversion of intervals to pandas DataFrame"""
        intervals_list = [(0, 100), (200, 260)]
        event_labels = ["stim", "resp"]
        event_indices = np.array([0, 1])
        event_onsets = [0, 200]
        intervals = Intervals(
            intervals_list,
            units="ms",
            label="events",
            event_labels=event_labels,
            event_indices=event_indices,
            event_onsets=event_onsets,
        )

        df = intervals.as_pandas()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertTrue((df["start"].values == np.array([0, 200])).all())
        self.assertTrue((df["duration"].values == np.array([100, 60])).all())
        self.assertTrue((df["event_label"].values == np.array(event_labels)).all())
        self.assertTrue((df["event_index"].values == event_indices).all())
        self.assertTrue((df["event_onset"].values == np.array(event_onsets)).all())

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
        event_indices = [0, 1, 2]
        event_onsets = [10, 60, 210]
        
        intervals = Intervals(intervals_list, units="ms", label="test",
                             event_labels=event_labels,
                             event_indices=event_indices,
                             event_onsets=event_onsets)
        
        merged = intervals.merge()
        
        # Should have 2 intervals (first two merged, third separate)
        self.assertEqual(len(merged), 2)
        
        # First interval: merged labels with underscore
        self.assertEqual(merged.event_labels[0], "event1_event2")
        # Second interval: unchanged label
        self.assertEqual(merged.event_labels[1], "event3")
        
        # First interval: first index from merged group
        self.assertEqual(merged.event_indices[0], 0)
        # Second interval: original index
        self.assertEqual(merged.event_indices[1], 2)
        
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
    
    def test_intervals_stats(self):
        """Test getting interval statistics"""
        intervals_list = [(0, 100), (0, 200), (0, 300)]
        intervals = Intervals(intervals_list, units="ms")
        
        stats = intervals.stats()
        self.assertIsInstance(stats, IntervalStats)
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["mean"], 200.0)  # (100+200+300)/3
        self.assertEqual(stats["min"], 100.0)
        self.assertEqual(stats["max"], 300.0)
    
    def test_intervals_repr(self):
        """Test string representation of Intervals"""
        intervals_list = [(0, 100), (0, 200), (0, 300)]
        intervals = Intervals(intervals_list, units="ms", label="test_intervals")
        
        repr_str = repr(intervals)
        self.assertIn("test_intervals", repr_str)
        self.assertIn("3 intervals", repr_str)
        self.assertIn("units=ms", repr_str)
    
    def test_intervals_repr_empty(self):
        """Test string representation of empty Intervals"""
        intervals = Intervals([], units="ms", label="empty")
        
        repr_str = repr(intervals)
        self.assertIn("empty", repr_str)
        self.assertIn("0 intervals", repr_str)
    
    def test_intervals_repr_no_label(self):
        """Test string representation without label"""
        intervals_list = [(0, 100)]
        intervals = Intervals(intervals_list, units=None)
        
        repr_str = repr(intervals)
        self.assertIn("Intervals", repr_str)
        self.assertIn("units=None", repr_str)


class TestIntervalsWithGetIntervals(unittest.TestCase):
    """Test Intervals class with get_intervals() method"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
    
    def test_get_intervals_returns_intervals_object(self):
        """Test that get_intervals returns Intervals object"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        self.assertIsInstance(intervals, Intervals)
        self.assertEqual(intervals.units, "ms")
        self.assertIsNotNone(intervals.label)
    
    def test_get_intervals_has_metadata(self):
        """Test that Intervals from get_intervals has proper metadata"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        self.assertIsNotNone(intervals.event_labels)
        self.assertIsNotNone(intervals.event_indices)
        self.assertGreater(len(intervals), 0)
    
    def test_get_intervals_different_units(self):
        """Test get_intervals with different units"""
        for units in ["ms", "sec", "min", None]:
            intervals = self.data.get_intervals("F", interval=(-200, 200), units=units)
            self.assertIsInstance(intervals, Intervals)
            self.assertEqual(intervals.units, units)
    
    def test_intervals_label_from_string(self):
        """Test that label is set from string event_select"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        self.assertEqual(intervals.label, "F")
    
    def test_intervals_can_be_iterated(self):
        """Test that Intervals from get_intervals can be iterated"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
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
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms", label=custom_label)
        
        self.assertEqual(intervals.label, custom_label)
    
    def test_intervals_automatic_label_when_none(self):
        """Test that automatic label is used when label parameter is None"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms", label=None)
        
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
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        fig = plt.figure()
        intervals.plot()
        
        # Should have created axes with content
        ax = plt.gca()
        self.assertIsNotNone(ax)
        self.assertGreater(len(ax.get_lines()), 0)
    
    def test_plot_with_labels(self):
        """Test that plot shows labels when requested"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        fig = plt.figure()
        intervals.plot(show_labels=True)
        
        ax = plt.gca()
        # Should have text labels
        texts = ax.texts
        self.assertGreater(len(texts), 0)
    
    def test_plot_without_labels(self):
        """Test that plot hides labels when requested"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        fig = plt.figure()
        intervals.plot(show_labels=False)
        
        ax = plt.gca()
        # Should not have text labels
        texts = ax.texts
        self.assertEqual(len(texts), 0)
    
    def test_plot_uses_current_axes(self):
        """Test that plot uses the current axes"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # Plot to first axes
        plt.sca(ax1)
        intervals.plot()
        
        # First axes should have lines, second should be empty
        self.assertGreater(len(ax1.get_lines()), 0)
        self.assertEqual(len(ax2.get_lines()), 0)
    
    def test_plot_sets_title_from_label(self):
        """Test that plot uses intervals label as title"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms", label="test_label")
        
        fig = plt.figure()
        intervals.plot()
        
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "test_label")
    
    def test_plot_sets_xlabel_from_units(self):
        """Test that plot sets x-axis label based on units"""
        # Test with ms
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        fig = plt.figure()
        intervals.plot()
        ax = plt.gca()
        self.assertIn("ms", ax.get_xlabel())
        plt.close('all')
        
        # Test with None (indices)
        intervals = self.data.get_intervals("F", interval=(-200, 200), units=None)
        fig = plt.figure()
        intervals.plot()
        ax = plt.gca()
        self.assertIn("indices", ax.get_xlabel())
    
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
        for units in ["ms", "sec", "min", None]:
            intervals = self.data.get_intervals("F", interval=(-200, 200), units=units)
            
            fig = plt.figure()
            intervals.plot()
            
            ax = plt.gca()
            # Should have created content
            self.assertGreater(len(ax.get_lines()), 0)
            
            plt.close('all')
    
    def test_plot_lines_are_black(self):
        """Test that interval lines are black"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        fig = plt.figure()
        intervals.plot()
        
        ax = plt.gca()
        lines = ax.get_lines()
        
        # Check that all lines are black (Intervals.plot doesn't have zero line)
        for line in lines:
            color = line.get_color()
            # Black is (0, 0, 0) or 'black' or 'k'
            self.assertIn(color, ['black', 'k', (0.0, 0.0, 0.0, 1.0)])


class TestIntervalsConversionMethods(unittest.TestCase):
    """Test Intervals conversion methods (as_index, to_units, __array__)"""
    
    def setUp(self):
        """Create test data"""
        self.data = pp.get_example_data("rlmw_002_short")
    
    def test_array_conversion_with_units_none(self):
        """Test __array__() with index units"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units=None)
        arr = np.array(intervals)
        
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape[1], 2)  # Should have 2 columns (start, end)
        self.assertEqual(len(arr), len(intervals))
    
    def test_array_conversion_with_units_ms(self):
        """Test __array__() with time units"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        arr = np.array(intervals)
        
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape[1], 2)
        self.assertEqual(len(arr), len(intervals))
    
    def test_as_index_with_units_none(self):
        """Test as_index() when intervals already in indices"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units=None)
        indices = intervals.as_index(self.data)
        
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.dtype, np.int_)
        self.assertEqual(indices.shape[1], 2)
        self.assertEqual(len(indices), len(intervals))
    
    def test_as_index_with_units_ms(self):
        """Test as_index() with millisecond units"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        indices = intervals.as_index(self.data)
        
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.dtype, np.int_)
        self.assertEqual(indices.shape[1], 2)
        
        # Check that indices are valid for the data
        for start, end in indices:
            self.assertGreaterEqual(start, 0)
            self.assertLess(end, len(self.data.tx))
    
    def test_as_index_with_units_sec(self):
        """Test as_index() with second units"""
        intervals = self.data.get_intervals("F", interval=(-0.2, 0.2), units="sec")
        indices = intervals.as_index(self.data)
        
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.dtype, np.int_)
        
        # Check that indices are valid
        for start, end in indices:
            self.assertGreaterEqual(start, 0)
            self.assertLess(end, len(self.data.tx))
    
    def test_to_units_ms_to_sec(self):
        """Test converting from milliseconds to seconds"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        intervals_sec = intervals.to_units("sec")
        
        self.assertEqual(intervals_sec.units, "sec")
        self.assertEqual(len(intervals_sec), len(intervals))
        
        # Check conversion accuracy (200ms = 0.2sec)
        arr_ms = np.array(intervals)
        arr_sec = np.array(intervals_sec)
        np.testing.assert_array_almost_equal(arr_sec, arr_ms / 1000.0, decimal=6)
    
    def test_to_units_sec_to_ms(self):
        """Test converting from seconds to milliseconds"""
        intervals = self.data.get_intervals("F", interval=(-0.2, 0.2), units="sec")
        intervals_ms = intervals.to_units("ms")
        
        self.assertEqual(intervals_ms.units, "ms")
        
        # Check conversion accuracy
        arr_sec = np.array(intervals)
        arr_ms = np.array(intervals_ms)
        np.testing.assert_array_almost_equal(arr_ms, arr_sec * 1000.0, decimal=6)
    
    def test_to_units_ms_to_min(self):
        """Test converting from milliseconds to minutes"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        intervals_min = intervals.to_units("min")
        
        self.assertEqual(intervals_min.units, "min")
        
        # Check conversion accuracy (200ms = 0.2/60 min)
        arr_ms = np.array(intervals)
        arr_min = np.array(intervals_min)
        np.testing.assert_array_almost_equal(arr_min, arr_ms / 60000.0, decimal=9)
    
    def test_to_units_same_units(self):
        """Test converting to same units returns self"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        intervals_same = intervals.to_units("ms")
        
        # Should return the same object
        self.assertIs(intervals_same, intervals)
    
    def test_to_units_with_aliases(self):
        """Test converting to units using aliases"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
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
    
    def test_to_units_from_none_raises_error(self):
        """Test that converting from indices raises error"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units=None)
        
        with self.assertRaises(ValueError) as cm:
            intervals.to_units("ms")
        
        self.assertIn("indices", str(cm.exception).lower())
    
    def test_to_units_to_none_raises_error(self):
        """Test that converting to indices raises error"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        with self.assertRaises(ValueError) as cm:
            intervals.to_units(None)
        
        self.assertIn("indices", str(cm.exception).lower())
    
    def test_to_units_unknown_source_raises_error(self):
        """Test that unknown source units raise error in constructor"""
        # Now that units are normalized in __init__, invalid units are caught there
        with self.assertRaises(ValueError) as cm:
            intervals = Intervals([(0, 100), (200, 300)], units="unknown", label="test")
        
        self.assertIn("unknown", str(cm.exception).lower())
    
    def test_to_units_unknown_target_raises_error(self):
        """Test that unknown target units raise error"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        with self.assertRaises(ValueError) as cm:
            intervals.to_units("unknown")
        
        self.assertIn("unknown", str(cm.exception).lower())
    
    def test_to_units_preserves_metadata(self):
        """Test that to_units preserves metadata"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        intervals_sec = intervals.to_units("sec")
        
        # Check that metadata is preserved
        self.assertEqual(intervals_sec.label, intervals.label)
        if intervals.event_labels is not None:
            self.assertEqual(len(intervals_sec.event_labels), len(intervals.event_labels))


class TestMergeIntervals(unittest.TestCase):
    """Test merge_intervals function"""
    
    def test_merge_intervals_from_args(self):
        """Test merging multiple Intervals passed as arguments"""
        intervals1 = Intervals([(0, 100), (200, 300)], units="ms", label="int1",
                              event_labels=["a", "b"])
        intervals2 = Intervals([(50, 150), (400, 500)], units="ms", label="int2",
                              event_labels=["c", "d"])
        
        merged = merge_intervals(intervals1, intervals2)
        
        self.assertIsInstance(merged, Intervals)
        self.assertEqual(len(merged), 4)  # All 4 intervals, not merged
        self.assertEqual(merged.units, "ms")
        self.assertEqual(merged.label, "merged")
        
        # Check metadata preserved
        self.assertIsNotNone(merged.event_labels)
        self.assertEqual(merged.event_labels, ["a", "b", "c", "d"])
    
    def test_merge_intervals_from_list(self):
        """Test merging from a list of Intervals"""
        intervals1 = Intervals([(0, 100)], units="ms", event_labels=["x"])
        intervals2 = Intervals([(200, 300)], units="ms", event_labels=["y"])
        intervals3 = Intervals([(400, 500)], units="ms", event_labels=["z"])
        
        intervals_list = [intervals1, intervals2, intervals3]
        merged = merge_intervals(intervals_list)
        
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged.event_labels, ["x", "y", "z"])
    
    def test_merge_intervals_from_dict(self):
        """Test merging from a dict of Intervals"""
        intervals1 = Intervals([(0, 100)], units="ms", event_labels=["p"])
        intervals2 = Intervals([(200, 300)], units="ms", event_labels=["q"])
        
        intervals_dict = {"left_pupil": intervals1, "right_pupil": intervals2}
        merged = merge_intervals(intervals_dict)
        
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.event_labels, ["p", "q"])
    
    def test_merge_intervals_custom_label(self):
        """Test merging with custom label"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(200, 300)], units="ms")
        
        merged = merge_intervals(intervals1, intervals2, label="custom")
        
        self.assertEqual(merged.label, "custom")
    
    def test_merge_intervals_preserves_all_metadata(self):
        """Test that all metadata is preserved"""
        intervals1 = Intervals([(0, 100), (200, 300)], units="ms",
                              event_labels=["a", "b"],
                              event_indices=[0, 1],
                              event_onsets=[10, 20])
        intervals2 = Intervals([(400, 500)], units="ms",
                              event_labels=["c"],
                              event_indices=[2],
                              event_onsets=[30])
        
        merged = merge_intervals(intervals1, intervals2)
        
        self.assertEqual(merged.event_labels, ["a", "b", "c"])
        self.assertEqual(list(merged.event_indices), [0, 1, 2])
        self.assertEqual(list(merged.event_onsets), [10, 20, 30])
    
    def test_merge_intervals_partial_metadata(self):
        """Test merging when only some objects have metadata"""
        intervals1 = Intervals([(0, 100)], units="ms",
                              event_labels=["a"])
        intervals2 = Intervals([(200, 300)], units="ms")  # No metadata
        intervals3 = Intervals([(400, 500)], units="ms",
                              event_labels=["c"])
        
        merged = merge_intervals(intervals1, intervals2, intervals3)
        
        # Should preserve labels with None for missing
        self.assertEqual(merged.event_labels, ["a", None, "c"])
    
    def test_merge_intervals_no_metadata(self):
        """Test merging when no objects have metadata"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(200, 300)], units="ms")
        
        merged = merge_intervals(intervals1, intervals2)
        
        # Should have no metadata
        self.assertIsNone(merged.event_labels)
        self.assertIsNone(merged.event_indices)
        self.assertIsNone(merged.event_onsets)
    
    def test_merge_intervals_does_not_merge_overlaps(self):
        """Test that overlapping intervals are NOT merged, just concatenated"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(50, 150)], units="ms")  # Overlaps with first
        
        merged = merge_intervals(intervals1, intervals2)
        
        # Should have both intervals, not merged
        self.assertEqual(len(merged), 2)
        arr = merged.to_array()
        self.assertEqual(arr[0, 0], 0)
        self.assertEqual(arr[0, 1], 100)
        self.assertEqual(arr[1, 0], 50)
        self.assertEqual(arr[1, 1], 150)
    
    def test_merge_intervals_then_merge_method(self):
        """Test combining intervals then merging overlaps with .merge()"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(50, 150)], units="ms")
        
        combined = merge_intervals(intervals1, intervals2)
        self.assertEqual(len(combined), 2)
        
        # Now merge overlaps
        merged = combined.merge()
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0], (0, 150))
    
    def test_merge_intervals_empty_intervals(self):
        """Test merging when one or more Intervals are empty"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([], units="ms")
        intervals3 = Intervals([(200, 300)], units="ms")
        
        merged = merge_intervals(intervals1, intervals2, intervals3)
        
        self.assertEqual(len(merged), 2)
    
    def test_merge_intervals_all_empty(self):
        """Test merging when all Intervals are empty"""
        intervals1 = Intervals([], units="ms")
        intervals2 = Intervals([], units="ms")
        
        merged = merge_intervals(intervals1, intervals2)
        
        self.assertEqual(len(merged), 0)
        self.assertEqual(merged.units, "ms")
    
    def test_merge_intervals_units_must_match(self):
        """Test that merging requires matching units"""
        intervals1 = Intervals([(0, 100)], units="ms")
        intervals2 = Intervals([(0, 1)], units="sec")
        
        with self.assertRaises(ValueError) as cm:
            merge_intervals(intervals1, intervals2)
        
        self.assertIn("units", str(cm.exception).lower())
    
    def test_merge_intervals_single_object(self):
        """Test merging with a single Intervals object"""
        intervals = Intervals([(0, 100), (200, 300)], units="ms",
                             event_labels=["a", "b"])
        
        merged = merge_intervals(intervals)
        
        # Should return a new object with same content
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.event_labels, ["a", "b"])
    
    def test_merge_intervals_preserves_data_time_range(self):
        """Test that data_time_range is preserved from first object"""
        intervals1 = Intervals([(0, 100)], units="ms",
                              data_time_range=(0, 1000))
        intervals2 = Intervals([(200, 300)], units="ms",
                              data_time_range=(0, 2000))
        
        merged = merge_intervals(intervals1, intervals2)
        
        # Should use data_time_range from first object
        self.assertEqual(merged.data_time_range, (0, 1000))
    
    def test_merge_intervals_no_arguments_raises_error(self):
        """Test that calling with no arguments raises error"""
        with self.assertRaises(ValueError) as cm:
            merge_intervals()
        
        self.assertIn("no intervals", str(cm.exception).lower())
    
    def test_merge_intervals_invalid_type_raises_error(self):
        """Test that invalid argument types raise error"""
        intervals1 = Intervals([(0, 100)], units="ms")
        
        with self.assertRaises(TypeError):
            merge_intervals(intervals1, "not an intervals object")
    
    def test_merge_intervals_different_units(self):
        """Test merging with different unit types"""
        for units in ["ms", "sec", "min", "h", None]:
            intervals1 = Intervals([(0, 100)], units=units)
            intervals2 = Intervals([(200, 300)], units=units)
            
            merged = merge_intervals(intervals1, intervals2)
            
            self.assertEqual(merged.units, units)
            self.assertEqual(len(merged), 2)


if __name__ == '__main__':
    unittest.main()

