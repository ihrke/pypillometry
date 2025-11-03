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
        
        # Check that lines are black
        for line in lines:
            color = line.get_color()
            # Black is (0, 0, 0) or 'black' or 'k'
            self.assertIn(color, ['black', 'k', (0.0, 0.0, 0.0, 1.0)])


if __name__ == '__main__':
    unittest.main()

