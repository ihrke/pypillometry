"""
Tests for the Intervals class.
"""
import unittest
import sys
sys.path.insert(0, "..")
import pypillometry as pp
import numpy as np
from pypillometry.intervals import Intervals, IntervalStats


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


if __name__ == '__main__':
    unittest.main()

