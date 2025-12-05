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
from pypillometry.intervals import Intervals, merge_intervals


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
        self.assertIsInstance(stats, dict)
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
    
    def test_add_preserves_event_indices(self):
        """Test that event_indices are concatenated"""
        a = Intervals([(0, 100)], units="ms", event_indices=np.array([0]))
        b = Intervals([(200, 300)], units="ms", event_indices=np.array([5]))
        c = a + b
        
        np.testing.assert_array_equal(c.event_indices, [0, 5])
    
    def test_add_preserves_event_onsets(self):
        """Test that event_onsets are concatenated and converted"""
        a = Intervals([(0, 100)], units="ms", event_onsets=np.array([50]))
        b = Intervals([(200, 300)], units="ms", event_onsets=np.array([250]))
        c = a + b
        
        np.testing.assert_array_equal(c.event_onsets, [50, 250])
    
    def test_add_expands_data_time_range(self):
        """Test that data_time_range is expanded to cover both"""
        a = Intervals([(0, 100)], units="ms", data_time_range=(0, 500))
        b = Intervals([(200, 300)], units="ms", data_time_range=(100, 600))
        c = a + b
        
        self.assertEqual(c.data_time_range, (0, 600))
    
    def test_add_partial_metadata_left_only(self):
        """Test adding when only left operand has metadata"""
        a = Intervals([(0, 100)], units="ms", event_labels=["a"])
        b = Intervals([(200, 300)], units="ms")
        c = a + b
        
        self.assertEqual(c.event_labels, ["a"])
    
    def test_add_partial_metadata_right_only(self):
        """Test adding when only right operand has metadata"""
        a = Intervals([(0, 100)], units="ms")
        b = Intervals([(200, 300)], units="ms", event_labels=["b"])
        c = a + b
        
        self.assertEqual(c.event_labels, ["b"])
    
    def test_add_no_metadata(self):
        """Test adding when neither operand has metadata"""
        a = Intervals([(0, 100)], units="ms")
        b = Intervals([(200, 300)], units="ms")
        c = a + b
        
        self.assertIsNone(c.event_labels)
        self.assertIsNone(c.event_indices)
        self.assertIsNone(c.event_onsets)
        self.assertIsNone(c.data_time_range)
    
    def test_add_indices_units(self):
        """Test adding intervals with units=None (indices)"""
        a = Intervals([(0, 100), (200, 300)], units=None)
        b = Intervals([(500, 600)], units=None)
        c = a + b
        
        self.assertEqual(len(c), 3)
        self.assertIsNone(c.units)
    
    def test_add_incompatible_units_raises_error(self):
        """Test that mixing time units and indices raises error"""
        a = Intervals([(0, 100)], units="ms")
        b = Intervals([(200, 300)], units=None)
        
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
    
    def test_add_with_unit_conversion_preserves_data_time_range(self):
        """Test that data_time_range is converted when units differ"""
        a = Intervals([(0, 1)], units="sec", data_time_range=(0, 10))
        b = Intervals([(2000, 3000)], units="ms", data_time_range=(0, 5000))
        c = a + b
        
        # b's range (0, 5000ms) should be converted to (0, 5sec)
        # Combined range should be (0, 10) expanded to include (0, 5) = (0, 10)
        self.assertEqual(c.data_time_range, (0, 10))


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
    
    def test_plot_accepts_color_kwarg(self):
        """Test that interval lines can be colored via kwargs"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        fig = plt.figure()
        intervals.plot(color='red')
        
        ax = plt.gca()
        lines = ax.get_lines()
        
        # Check that all lines are red (as specified)
        for line in lines:
            color = line.get_color()
            self.assertEqual(color, 'red')


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
    
    def test_to_indices_with_units_none(self):
        """Test to_units('indices') when intervals already in indices"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units=None)
        indices_intervals = intervals.to_units("indices")
        indices = np.array(indices_intervals).astype(int)
        
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.dtype, np.int_)
        self.assertEqual(indices.shape[1], 2)
        self.assertEqual(len(indices), len(intervals))
    
    def test_to_indices_with_units_ms(self):
        """Test to_units('indices') with millisecond units"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        indices_intervals = intervals.to_units("indices")
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
        intervals = self.data.get_intervals("F", interval=(-0.2, 0.2), units="sec")
        indices_intervals = intervals.to_units("indices")
        indices = np.array(indices_intervals).astype(int)
        
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
    
    def test_to_units_from_indices_works_with_sampling_rate(self):
        """Test that converting from indices works when sampling_rate is set"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units=None)
        # Should have sampling_rate set from get_intervals
        self.assertIsNotNone(intervals.sampling_rate)
        
        # Conversion should work
        intervals_ms = intervals.to_units("ms")
        self.assertEqual(intervals_ms.units, "ms")
    
    def test_to_units_from_indices_raises_error_without_sampling_rate(self):
        """Test that converting from indices raises error when sampling_rate not set"""
        intervals = Intervals([(0, 100), (200, 300)], units=None)
        
        with self.assertRaises(ValueError) as cm:
            intervals.to_units("ms")
        
        self.assertIn("sampling_rate", str(cm.exception).lower())
    
    def test_to_units_to_indices_works_with_sampling_rate(self):
        """Test that converting to indices works when sampling_rate is set"""
        intervals = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        # Should have sampling_rate set from get_intervals
        self.assertIsNotNone(intervals.sampling_rate)
        
        # Conversion should work
        intervals_idx = intervals.to_units("indices")
        self.assertIsNone(intervals_idx.units)
    
    def test_to_units_to_indices_raises_error_without_sampling_rate(self):
        """Test that converting to indices raises error when sampling_rate not set"""
        intervals = Intervals([(0, 100), (200, 300)], units="ms")
        
        with self.assertRaises(ValueError) as cm:
            intervals.to_units("indices")
        
        self.assertIn("sampling_rate", str(cm.exception).lower())
    
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
        arr = np.array(merged)
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


class TestIntervalsMaskConversion(unittest.TestCase):
    """Test Intervals to_mask() and from_mask() methods"""
    
    def test_to_mask_basic(self):
        """Test basic conversion to mask"""
        intervals = Intervals([(0, 10), (20, 30)], units=None, data_time_range=(0, 50))
        mask = intervals.to_mask()
        
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(len(mask), 50)
        self.assertEqual(mask.dtype, np.int8)
        
        # Check mask values
        np.testing.assert_array_equal(mask[0:10], 1)
        np.testing.assert_array_equal(mask[10:20], 0)
        np.testing.assert_array_equal(mask[20:30], 1)
        np.testing.assert_array_equal(mask[30:50], 0)
    
    def test_to_mask_with_explicit_length(self):
        """Test to_mask with explicit length argument"""
        intervals = Intervals([(0, 10), (20, 30)], units=None)
        mask = intervals.to_mask(length=40)
        
        self.assertEqual(len(mask), 40)
        np.testing.assert_array_equal(mask[0:10], 1)
        np.testing.assert_array_equal(mask[20:30], 1)
    
    def test_to_mask_empty_intervals(self):
        """Test to_mask with empty intervals"""
        intervals = Intervals([], units=None, data_time_range=(0, 100))
        mask = intervals.to_mask()
        
        self.assertEqual(len(mask), 100)
        np.testing.assert_array_equal(mask, 0)
    
    def test_to_mask_auto_converts_time_units(self):
        """Test that to_mask auto-converts time-based intervals to indices"""
        # 100ms intervals at 1000Hz = 100 samples
        intervals = Intervals([(0, 100), (200, 300)], units="ms", 
                             data_time_range=(0, 500), sampling_rate=1000)
        mask = intervals.to_mask()
        
        self.assertEqual(len(mask), 500)  # 500ms at 1000Hz = 500 samples
        np.testing.assert_array_equal(mask[0:100], 1)
        np.testing.assert_array_equal(mask[100:200], 0)
        np.testing.assert_array_equal(mask[200:300], 1)
        np.testing.assert_array_equal(mask[300:500], 0)
    
    def test_to_mask_raises_without_sampling_rate(self):
        """Test that to_mask raises error for time units without sampling_rate"""
        intervals = Intervals([(0, 100)], units="ms", data_time_range=(0, 500))
        
        with self.assertRaises(ValueError) as cm:
            intervals.to_mask()
        
        self.assertIn("sampling_rate", str(cm.exception).lower())
    
    def test_to_mask_raises_without_length_or_data_time_range(self):
        """Test that to_mask raises error without length or data_time_range"""
        intervals = Intervals([(0, 10)], units=None)
        
        with self.assertRaises(ValueError) as cm:
            intervals.to_mask()
        
        self.assertIn("length", str(cm.exception).lower())
    
    def test_to_mask_clips_to_bounds(self):
        """Test that intervals outside mask bounds are clipped"""
        intervals = Intervals([(-10, 10), (40, 60)], units=None, data_time_range=(0, 50))
        mask = intervals.to_mask()
        
        # Should clip to [0, 50)
        np.testing.assert_array_equal(mask[0:10], 1)
        np.testing.assert_array_equal(mask[40:50], 1)
    
    def test_from_mask_basic(self):
        """Test basic creation from mask"""
        mask = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        intervals = Intervals.from_mask(mask)
        
        self.assertIsInstance(intervals, Intervals)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals.intervals, [(2, 5), (7, 9)])
        self.assertIsNone(intervals.units)
        self.assertEqual(intervals.data_time_range, (0, 10))
    
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
    
    def test_from_mask_with_sampling_rate(self):
        """Test from_mask with sampling_rate"""
        mask = np.array([1, 1, 0, 0, 1, 1])
        intervals = Intervals.from_mask(mask, sampling_rate=1000)
        
        self.assertEqual(intervals.sampling_rate, 1000)
    
    def test_from_mask_boolean_input(self):
        """Test from_mask with boolean array"""
        mask = np.array([False, True, True, False, True])
        intervals = Intervals.from_mask(mask)
        
        self.assertEqual(intervals.intervals, [(1, 3), (4, 5)])
    
    def test_roundtrip_to_mask_from_mask(self):
        """Test roundtrip: intervals -> mask -> intervals"""
        original = Intervals([(10, 20), (30, 50), (70, 80)], units=None, data_time_range=(0, 100))
        
        mask = original.to_mask()
        recovered = Intervals.from_mask(mask)
        
        self.assertEqual(original.intervals, recovered.intervals)


class TestIntervalsSubtractOperator(unittest.TestCase):
    """Test Intervals - operator for subtracting intervals"""
    
    def test_subtract_basic(self):
        """Test basic subtraction of intervals"""
        a = Intervals([(0, 100), (200, 300)], units=None, data_time_range=(0, 400))
        b = Intervals([(50, 150)], units=None, data_time_range=(0, 400))
        c = a - b
        
        self.assertIsInstance(c, Intervals)
        # (0, 100) - (50, 150) = (0, 50)
        # (200, 300) unchanged
        self.assertEqual(c.intervals, [(0, 50), (200, 300)])
    
    def test_subtract_complete_removal(self):
        """Test subtraction that completely removes an interval"""
        a = Intervals([(50, 100)], units=None, data_time_range=(0, 200))
        b = Intervals([(0, 150)], units=None, data_time_range=(0, 200))
        c = a - b
        
        # (50, 100) is completely contained in (0, 150), so nothing remains
        self.assertEqual(len(c), 0)
    
    def test_subtract_no_overlap(self):
        """Test subtraction with no overlap"""
        a = Intervals([(0, 50), (100, 150)], units=None, data_time_range=(0, 200))
        b = Intervals([(60, 90)], units=None, data_time_range=(0, 200))
        c = a - b
        
        # No overlap, a unchanged
        self.assertEqual(c.intervals, [(0, 50), (100, 150)])
    
    def test_subtract_splits_interval(self):
        """Test subtraction that splits an interval in two"""
        a = Intervals([(0, 100)], units=None, data_time_range=(0, 200))
        b = Intervals([(40, 60)], units=None, data_time_range=(0, 200))
        c = a - b
        
        # (0, 100) - (40, 60) = (0, 40) and (60, 100)
        self.assertEqual(c.intervals, [(0, 40), (60, 100)])
    
    def test_subtract_empty_left(self):
        """Test subtraction with empty left operand"""
        a = Intervals([], units=None, data_time_range=(0, 100))
        b = Intervals([(20, 50)], units=None, data_time_range=(0, 100))
        c = a - b
        
        self.assertEqual(len(c), 0)
    
    def test_subtract_empty_right(self):
        """Test subtraction with empty right operand"""
        a = Intervals([(0, 50)], units=None, data_time_range=(0, 100))
        b = Intervals([], units=None, data_time_range=(0, 100))
        c = a - b
        
        self.assertEqual(c.intervals, [(0, 50)])
    
    def test_subtract_with_time_units(self):
        """Test subtraction with time-based intervals"""
        # 1000Hz sampling rate
        a = Intervals([(0, 100), (200, 300)], units="ms", 
                     data_time_range=(0, 400), sampling_rate=1000)
        b = Intervals([(50, 150)], units="ms", 
                     data_time_range=(0, 400), sampling_rate=1000)
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
        a = Intervals([(0, 100)], units=None, data_time_range=(0, 200), label="a")
        b = Intervals([(50, 75)], units=None, data_time_range=(0, 200), label="b")
        c = a - b
        
        self.assertIn("a", c.label)
        self.assertIn("b", c.label)
    
    def test_subtract_preserves_sampling_rate(self):
        """Test that subtraction preserves sampling_rate"""
        a = Intervals([(0, 100)], units=None, data_time_range=(0, 200), sampling_rate=500)
        b = Intervals([(50, 75)], units=None, data_time_range=(0, 200))
        c = a - b
        
        self.assertEqual(c.sampling_rate, 500)
    
    def test_subtract_non_intervals_returns_notimplemented(self):
        """Test that subtracting non-Intervals returns NotImplemented"""
        a = Intervals([(0, 100)], units=None, data_time_range=(0, 200))
        
        result = a.__sub__("not an Intervals")
        self.assertEqual(result, NotImplemented)
    
    def test_subtract_raises_without_data_time_range(self):
        """Test that subtraction raises error without data_time_range"""
        a = Intervals([(0, 100)], units=None)
        b = Intervals([(50, 75)], units=None)
        
        with self.assertRaises(ValueError) as cm:
            a - b
        
        self.assertIn("data_time_range", str(cm.exception).lower())
    
    def test_subtract_uses_other_data_time_range(self):
        """Test that subtraction uses other's data_time_range if self doesn't have one"""
        a = Intervals([(0, 100)], units=None)
        b = Intervals([(50, 75)], units=None, data_time_range=(0, 200))
        c = a - b
        
        # Should work using b's data_time_range
        self.assertEqual(c.intervals, [(0, 50), (75, 100)])
    
    def test_subtract_multiple_from_multiple(self):
        """Test subtracting multiple intervals from multiple intervals"""
        a = Intervals([(0, 50), (100, 150), (200, 250)], units=None, data_time_range=(0, 300))
        b = Intervals([(25, 75), (125, 225)], units=None, data_time_range=(0, 300))
        c = a - b
        
        # (0, 50) - (25, 75) = (0, 25)
        # (100, 150) - (125, 225) = (100, 125)
        # (200, 250) - (125, 225) = (225, 250)
        self.assertEqual(c.intervals, [(0, 25), (100, 125), (225, 250)])


if __name__ == '__main__':
    unittest.main()

