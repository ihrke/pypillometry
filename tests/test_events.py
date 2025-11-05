"""
Tests for the Events class.
"""
import unittest
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np
from pypillometry.events import Events


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


if __name__ == '__main__':
    unittest.main()

