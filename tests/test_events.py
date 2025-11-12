"""
Tests for the Events class.
"""
import unittest
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np
from pypillometry.events import Events
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestEventsClass(unittest.TestCase):
    """Test Events class basic functionality"""
    
    def test_events_creation_from_lists(self):
        """Test creating Events from lists"""
        onsets = [100, 500, 1000]
        labels = ["A", "B", "C"]
        events = Events(onsets, labels, units="ms")
        
        self.assertEqual(len(events), 3)
        self.assertEqual(events.units, "ms")
        np.testing.assert_array_equal(events.onsets, [100, 500, 1000])
        np.testing.assert_array_equal(events.labels, ["A", "B", "C"])
    
    def test_events_creation_from_arrays(self):
        """Test creating Events from numpy arrays"""
        onsets = np.array([1.0, 2.0, 3.0])
        labels = np.array(["X", "Y", "Z"])
        events = Events(onsets, labels, units="sec")
        
        self.assertEqual(len(events), 3)
        self.assertEqual(events.units, "sec")
        self.assertIsInstance(events.onsets, np.ndarray)
        self.assertIsInstance(events.labels, np.ndarray)
    
    def test_events_creation_with_time_range(self):
        """Test creating Events with data_time_range"""
        events = Events([100, 200], ["A", "B"], units="ms", data_time_range=(0, 1000))
        
        self.assertEqual(events.data_time_range, (0, 1000))
    
    def test_events_creation_mismatched_lengths_raises_error(self):
        """Test that mismatched onsets and labels raise error"""
        with self.assertRaises(ValueError) as cm:
            Events([100, 200], ["A"], units="ms")
        
        self.assertIn("same length", str(cm.exception))
    
    def test_events_creation_with_none_units(self):
        """Test creating Events with units=None (indices)"""
        events = Events([10, 20, 30], ["A", "B", "C"], units=None)
        
        self.assertEqual(events.units, None)
        self.assertEqual(len(events), 3)
    
    def test_events_empty(self):
        """Test creating empty Events"""
        events = Events([], [], units="ms")
        
        self.assertEqual(len(events), 0)
        self.assertEqual(events.units, "ms")
        np.testing.assert_array_equal(events.onsets, [])
        np.testing.assert_array_equal(events.labels, [])


class TestEventsAccessors(unittest.TestCase):
    """Test Events class accessor methods"""
    
    def setUp(self):
        """Create test events"""
        self.events = Events(
            onsets=[100, 500, 1000, 1500],
            labels=["stim1", "stim2", "resp1", "resp2"],
            units="ms"
        )
    
    def test_events_length(self):
        """Test len() on Events"""
        self.assertEqual(len(self.events), 4)
    
    def test_events_iteration(self):
        """Test iterating over Events"""
        result = list(self.events)
        
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], (100, "stim1"))
        self.assertEqual(result[1], (500, "stim2"))
        self.assertEqual(result[2], (1000, "resp1"))
        self.assertEqual(result[3], (1500, "resp2"))
    
    def test_events_indexing_int(self):
        """Test integer indexing Events"""
        self.assertEqual(self.events[0], (100, "stim1"))
        self.assertEqual(self.events[1], (500, "stim2"))
        self.assertEqual(self.events[-1], (1500, "resp2"))
    
    def test_events_indexing_slice(self):
        """Test slice indexing Events"""
        subset = self.events[1:3]
        
        self.assertIsInstance(subset, Events)
        self.assertEqual(len(subset), 2)
        np.testing.assert_array_equal(subset.onsets, [500, 1000])
        np.testing.assert_array_equal(subset.labels, ["stim2", "resp1"])
        self.assertEqual(subset.units, "ms")
    
    def test_events_indexing_invalid_type(self):
        """Test that invalid index type raises error"""
        with self.assertRaises(TypeError):
            _ = self.events["invalid"]
    
    def test_events_to_array(self):
        """Test converting Events to arrays"""
        onsets, labels = self.events.to_array()
        
        self.assertIsInstance(onsets, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        np.testing.assert_array_equal(onsets, [100, 500, 1000, 1500])
        np.testing.assert_array_equal(labels, ["stim1", "stim2", "resp1", "resp2"])
        
        # Check that it's a copy (not a reference)
        onsets[0] = 999
        self.assertEqual(self.events.onsets[0], 100)


class TestEventsUnitConversion(unittest.TestCase):
    """Test Events unit conversion functionality"""
    
    def test_to_units_ms_to_sec(self):
        """Test converting from milliseconds to seconds"""
        events = Events([1000, 2000, 3000], ["A", "B", "C"], units="ms")
        events_sec = events.to_units("sec")
        
        self.assertEqual(events_sec.units, "sec")
        self.assertEqual(len(events_sec), 3)
        np.testing.assert_array_almost_equal(events_sec.onsets, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(events_sec.labels, ["A", "B", "C"])
    
    def test_to_units_sec_to_ms(self):
        """Test converting from seconds to milliseconds"""
        events = Events([1.0, 2.5, 3.0], ["A", "B", "C"], units="sec")
        events_ms = events.to_units("ms")
        
        self.assertEqual(events_ms.units, "ms")
        np.testing.assert_array_almost_equal(events_ms.onsets, [1000, 2500, 3000])
    
    def test_to_units_ms_to_min(self):
        """Test converting from milliseconds to minutes"""
        events = Events([60000, 120000], ["A", "B"], units="ms")
        events_min = events.to_units("min")
        
        self.assertEqual(events_min.units, "min")
        np.testing.assert_array_almost_equal(events_min.onsets, [1.0, 2.0])
    
    def test_to_units_ms_to_h(self):
        """Test converting from milliseconds to hours"""
        events = Events([3600000, 7200000], ["A", "B"], units="ms")
        events_h = events.to_units("h")
        
        self.assertEqual(events_h.units, "h")
        np.testing.assert_array_almost_equal(events_h.onsets, [1.0, 2.0])
    
    def test_to_units_same_units_returns_copy(self):
        """Test converting to same units returns a copy"""
        events = Events([100, 200], ["A", "B"], units="ms")
        events_same = events.to_units("ms")
        
        # Should be a new object (not same reference)
        self.assertIsNot(events_same, events)
        # But with same values
        np.testing.assert_array_equal(events_same.onsets, events.onsets)
        self.assertEqual(events_same.units, events.units)
    
    def test_to_units_preserves_time_range(self):
        """Test that unit conversion also converts time range"""
        events = Events([1000, 2000], ["A", "B"], units="ms", data_time_range=(0, 10000))
        events_sec = events.to_units("sec")
        
        self.assertIsNotNone(events_sec.data_time_range)
        self.assertAlmostEqual(events_sec.data_time_range[0], 0.0)
        self.assertAlmostEqual(events_sec.data_time_range[1], 10.0)
    
    def test_to_units_from_none_raises_error(self):
        """Test that converting from indices (units=None) raises error"""
        events = Events([10, 20], ["A", "B"], units=None)
        
        with self.assertRaises(ValueError) as cm:
            events.to_units("ms")
        
        self.assertIn("indices", str(cm.exception))
    
    def test_to_units_to_none_raises_error(self):
        """Test that converting to indices (units=None) raises error"""
        events = Events([1000, 2000], ["A", "B"], units="ms")
        
        with self.assertRaises(ValueError) as cm:
            events.to_units(None)
        
        self.assertIn("indices", str(cm.exception))
    
    def test_to_units_unknown_source_raises_error(self):
        """Test that unknown source units raise error"""
        events = Events([100, 200], ["A", "B"], units="unknown")
        
        with self.assertRaises(ValueError) as cm:
            events.to_units("ms")
        
        self.assertIn("unknown", str(cm.exception).lower())
    
    def test_to_units_unknown_target_raises_error(self):
        """Test that unknown target units raise error"""
        events = Events([100, 200], ["A", "B"], units="ms")
        
        with self.assertRaises(ValueError) as cm:
            events.to_units("unknown")
        
        self.assertIn("unknown", str(cm.exception).lower())
    
    def test_to_units_chain_conversions(self):
        """Test chaining multiple unit conversions"""
        events = Events([3600000], ["A"], units="ms")
        
        # ms -> sec -> min -> h
        events_sec = events.to_units("sec")
        events_min = events_sec.to_units("min")
        events_h = events_min.to_units("h")
        
        self.assertAlmostEqual(events_h.onsets[0], 1.0)
        self.assertEqual(events_h.units, "h")


class TestEventsRepr(unittest.TestCase):
    """Test Events string representation"""
    
    def test_repr_with_events(self):
        """Test repr with multiple events"""
        events = Events([100, 200, 300], ["A", "B", "C"], units="ms")
        repr_str = repr(events)
        
        self.assertIn("3 events", repr_str)
        self.assertIn("units=ms", repr_str)
    
    def test_repr_single_event(self):
        """Test repr with single event"""
        events = Events([100], ["A"], units="sec")
        repr_str = repr(events)
        
        self.assertIn("1 event", repr_str)
        self.assertIn("units=sec", repr_str)
    
    def test_repr_empty_events(self):
        """Test repr with empty events"""
        events = Events([], [], units="ms")
        repr_str = repr(events)
        
        self.assertIn("0 events", repr_str)
    
    def test_repr_with_none_units(self):
        """Test repr with units=None"""
        events = Events([10, 20], ["A", "B"], units=None)
        repr_str = repr(events)
        
        self.assertIn("units=None", repr_str)
        self.assertIn("indices", repr_str)
    
    def test_repr_with_time_range(self):
        """Test repr with data_time_range"""
        events = Events([100, 200], ["A", "B"], units="ms", data_time_range=(0, 1000))
        repr_str = repr(events)
        
        self.assertIn("range=", repr_str)
        self.assertIn("0.0", repr_str)
        self.assertIn("1000.0", repr_str)
    
    def test_repr_without_time_range(self):
        """Test repr without data_time_range"""
        events = Events([100, 200], ["A", "B"], units="ms")
        repr_str = repr(events)
        
        # Should not have range= in the string
        self.assertNotIn("range=", repr_str)


class TestEventsEdgeCases(unittest.TestCase):
    """Test Events class edge cases"""
    
    def test_events_with_duplicate_labels(self):
        """Test Events with duplicate labels"""
        events = Events([100, 200, 300], ["A", "A", "B"], units="ms")
        
        self.assertEqual(len(events), 3)
        np.testing.assert_array_equal(events.labels, ["A", "A", "B"])
    
    def test_events_with_numeric_labels(self):
        """Test Events with numeric labels (converted to strings)"""
        events = Events([100, 200], [1, 2], units="ms")
        
        # Labels should be converted to strings
        self.assertIsInstance(events.labels[0], str)
        np.testing.assert_array_equal(events.labels, ["1", "2"])
    
    def test_events_with_negative_onsets(self):
        """Test Events with negative onset times"""
        events = Events([-100, 0, 100], ["A", "B", "C"], units="ms")
        
        self.assertEqual(len(events), 3)
        np.testing.assert_array_equal(events.onsets, [-100, 0, 100])
    
    def test_events_with_unsorted_onsets(self):
        """Test Events with unsorted onset times (should be allowed)"""
        events = Events([300, 100, 200], ["C", "A", "B"], units="ms")
        
        # Should preserve the order as given
        np.testing.assert_array_equal(events.onsets, [300, 100, 200])
        np.testing.assert_array_equal(events.labels, ["C", "A", "B"])
    
    def test_events_with_float_onsets(self):
        """Test Events with floating point onset times"""
        events = Events([100.5, 200.7, 300.9], ["A", "B", "C"], units="ms")
        
        self.assertEqual(len(events), 3)
        np.testing.assert_array_almost_equal(events.onsets, [100.5, 200.7, 300.9])
    
    def test_events_slice_preserves_time_range(self):
        """Test that slicing preserves data_time_range"""
        events = Events([100, 200, 300], ["A", "B", "C"], units="ms", data_time_range=(0, 1000))
        subset = events[1:3]
        
        self.assertEqual(subset.data_time_range, events.data_time_range)
    
    def test_events_with_empty_string_labels(self):
        """Test Events with empty string labels"""
        events = Events([100, 200], ["", "B"], units="ms")
        
        self.assertEqual(len(events), 2)
        self.assertEqual(events.labels[0], "")
        self.assertEqual(events.labels[1], "B")


class TestEventsFiltering(unittest.TestCase):
    """Test Events.filter() method"""
    
    def setUp(self):
        """Create test events"""
        self.events = Events(
            onsets=[100, 500, 1000, 1500, 2000],
            labels=["stim1", "resp", "stim2", "resp", "stim3"],
            units="ms",
            data_time_range=(0, 3000)
        )
    
    def test_filter_string_substring(self):
        """Test filtering by substring matching"""
        stim_events = self.events.filter("stim")
        
        self.assertIsInstance(stim_events, Events)
        self.assertEqual(len(stim_events), 3)
        np.testing.assert_array_equal(stim_events.onsets, [100, 1000, 2000])
        np.testing.assert_array_equal(stim_events.labels, ["stim1", "stim2", "stim3"])
    
    def test_filter_string_exact_match(self):
        """Test filtering with exact label match"""
        resp_events = self.events.filter("resp")
        
        self.assertEqual(len(resp_events), 2)
        np.testing.assert_array_equal(resp_events.onsets, [500, 1500])
        np.testing.assert_array_equal(resp_events.labels, ["resp", "resp"])
    
    def test_filter_string_no_match(self):
        """Test filtering with no matches returns empty Events"""
        empty_events = self.events.filter("nonexistent")
        
        self.assertIsInstance(empty_events, Events)
        self.assertEqual(len(empty_events), 0)
        self.assertEqual(empty_events.units, "ms")
    
    def test_filter_function_simple(self):
        """Test filtering with simple function"""
        # Filter events containing digit 2
        events = self.events.filter(lambda label: "2" in label)
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events.labels[0], "stim2")
    
    def test_filter_function_complex(self):
        """Test filtering with more complex function"""
        # Filter events that start with 's'
        events = self.events.filter(lambda label: label.startswith("s"))
        
        self.assertEqual(len(events), 3)
        np.testing.assert_array_equal(events.labels, ["stim1", "stim2", "stim3"])
    
    def test_filter_function_with_condition(self):
        """Test filtering with conditional function"""
        # Filter events where label length > 4
        events = self.events.filter(lambda label: len(label) > 4)
        
        self.assertEqual(len(events), 3)
        np.testing.assert_array_equal(events.labels, ["stim1", "stim2", "stim3"])
    
    def test_filter_time_range_middle(self):
        """Test filtering by time range in middle"""
        middle_events = self.events.filter((400, 1600))
        
        self.assertEqual(len(middle_events), 3)
        np.testing.assert_array_equal(middle_events.onsets, [500, 1000, 1500])
    
    def test_filter_time_range_beginning(self):
        """Test filtering by time range at beginning"""
        early_events = self.events.filter((0, 600))
        
        self.assertEqual(len(early_events), 2)
        np.testing.assert_array_equal(early_events.onsets, [100, 500])
    
    def test_filter_time_range_end(self):
        """Test filtering by time range at end"""
        late_events = self.events.filter((1400, 3000))
        
        self.assertEqual(len(late_events), 2)
        np.testing.assert_array_equal(late_events.onsets, [1500, 2000])
    
    def test_filter_time_range_inclusive(self):
        """Test that time range filtering is inclusive"""
        # Should include events exactly at boundaries
        events = self.events.filter((500, 1000))
        
        self.assertEqual(len(events), 2)
        np.testing.assert_array_equal(events.onsets, [500, 1000])
    
    def test_filter_time_range_no_match(self):
        """Test time range filtering with no matches"""
        empty_events = self.events.filter((5000, 6000))
        
        self.assertEqual(len(empty_events), 0)
    
    def test_filter_time_range_invalid_order(self):
        """Test that invalid time range raises error"""
        with self.assertRaises(ValueError) as cm:
            self.events.filter((1000, 500))
        
        self.assertIn("min_time must be < max_time", str(cm.exception))
    
    def test_filter_time_range_invalid_length(self):
        """Test that time range with wrong length raises error"""
        with self.assertRaises(ValueError) as cm:
            self.events.filter((100, 200, 300))
        
        self.assertIn("tuple of (min_time, max_time)", str(cm.exception))
    
    def test_filter_invalid_type(self):
        """Test that invalid selector type raises error"""
        with self.assertRaises(TypeError) as cm:
            self.events.filter(123)
        
        self.assertIn("str, callable, or tuple", str(cm.exception))
    
    def test_filter_preserves_units(self):
        """Test that filtering preserves units"""
        filtered = self.events.filter("stim")
        
        self.assertEqual(filtered.units, self.events.units)
    
    def test_filter_preserves_time_range(self):
        """Test that filtering preserves data_time_range"""
        filtered = self.events.filter("stim")
        
        self.assertEqual(filtered.data_time_range, self.events.data_time_range)
    
    def test_filter_chaining_string_then_time(self):
        """Test chaining filters: string then time range"""
        result = self.events.filter("stim").filter((0, 1500))
        
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result.labels, ["stim1", "stim2"])
    
    def test_filter_chaining_time_then_string(self):
        """Test chaining filters: time range then string"""
        result = self.events.filter((400, 1600)).filter("resp")
        
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result.onsets, [500, 1500])
    
    def test_filter_chaining_multiple(self):
        """Test chaining multiple filters"""
        result = (self.events
                  .filter(lambda label: len(label) > 4)
                  .filter((0, 1500))
                  .filter("1"))
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result.labels[0], "stim1")
    
    def test_filter_returns_new_object(self):
        """Test that filtering returns a new Events object (immutable)"""
        original_len = len(self.events)
        filtered = self.events.filter("stim")
        
        # Original should be unchanged
        self.assertEqual(len(self.events), original_len)
        self.assertIsNot(filtered, self.events)
    
    def test_filter_empty_events(self):
        """Test filtering empty Events"""
        empty = Events([], [], units="ms")
        result = empty.filter("test")
        
        self.assertEqual(len(result), 0)
    
    def test_filter_with_different_units(self):
        """Test filtering works with different units"""
        events_sec = Events([1.0, 2.0, 3.0], ["A", "B", "C"], units="sec")
        filtered = events_sec.filter((1.5, 2.5))
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.onsets[0], 2.0)
        self.assertEqual(filtered.units, "sec")


class TestEventsPlot(unittest.TestCase):
    """Tests for the Events.plot method."""

    def setUp(self):
        self.events = Events(
            onsets=np.linspace(0, 1000, 50),
            labels=[f"E{i}" for i in range(50)],
            units="ms"
        )

    def test_plot_auto_mode_spaced_labels(self):
        fig, ax = plt.subplots()
        try:
            returned_ax = self.events.plot(show_labels="auto", units="ms")
            self.assertIs(returned_ax, ax)
            texts = ax.texts
            self.assertGreater(len(texts), 0)
            xs = [txt.get_position()[0] for txt in texts]
            if len(xs) > 1:
                diffs = np.diff(xs)
                self.assertTrue(np.all(diffs >= 0))
        finally:
            plt.close(fig)

    def test_plot_all_labels(self):
        fig, ax = plt.subplots()
        try:
            returned_ax = self.events.plot(show_labels="all", units="ms")
            self.assertIs(returned_ax, ax)
            self.assertEqual(len(ax.texts), len(self.events))
        finally:
            plt.close(fig)

    def test_plot_no_labels(self):
        fig, ax = plt.subplots()
        try:
            returned_ax = self.events.plot(show_labels="none", units="ms")
            self.assertIs(returned_ax, ax)
            self.assertEqual(len(ax.texts), 0)
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()

