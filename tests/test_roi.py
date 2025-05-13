"""Test cases for ROI classes."""

import unittest
import numpy as np
from pypillometry.roi import ROI, CircularROI, RectangularROI

class TestROI(unittest.TestCase):
    """Test cases for base ROI class."""
    
    def test_base_roi_initialization(self):
        """Test initialization of base ROI class."""
        roi = ROI(name="test_roi")
        self.assertEqual(roi.name, "test_roi")
        
        roi = ROI()  # Test without name
        self.assertIsNone(roi.name)
    
    def test_base_roi_contains_not_implemented(self):
        """Test that base ROI class raises NotImplementedError for __contains__."""
        roi = ROI()
        with self.assertRaises(NotImplementedError):
            _ = (0, 0) in roi


class TestCircularROI(unittest.TestCase):
    """Test cases for CircularROI class."""
    
    def setUp(self):
        """Set up test cases."""
        self.circle = CircularROI(center=(0, 0), radius=5, name="test_circle")
    
    def test_initialization(self):
        """Test initialization of CircularROI."""
        self.assertEqual(self.circle.name, "test_circle")
        np.testing.assert_array_equal(self.circle.center, np.array([0, 0]))
        self.assertEqual(self.circle.radius, 5)
    
    def test_single_point_contains(self):
        """Test __contains__ with single points."""
        # Points inside circle
        self.assertTrue((0, 0) in self.circle)  # Center
        self.assertTrue((3, 4) in self.circle)  # On radius
        self.assertTrue((2, 2) in self.circle)  # Inside
        
        # Points outside circle
        self.assertFalse((6, 0) in self.circle)  # Outside
        self.assertFalse((0, 6) in self.circle)  # Outside
        self.assertFalse((4, 4) in self.circle)  # Outside
    
    def test_array_contains(self):
        """Test .contains() with arrays of points."""
        points = np.array([
            [0, 0],    # Center
            [3, 4],    # On radius
            [2, 2],    # Inside
            [6, 0],    # Outside
            [0, 6],    # Outside
            [4, 4]     # Outside
        ])
        
        expected = np.array([True, True, True, False, False, False])
        result = self.circle.contains(points)
        np.testing.assert_array_equal(result, expected)
        # __contains__ should raise ValueError for arrays
        with self.assertRaises(ValueError):
            _ = points in self.circle
    
    def test_edge_cases(self):
        """Test edge cases for CircularROI."""
        # Zero radius
        zero_circle = CircularROI(center=(0, 0), radius=0)
        self.assertTrue((0, 0) in zero_circle)
        self.assertFalse((0.1, 0) in zero_circle)
        
        # Negative radius (should raise ValueError)
        with self.assertRaises(ValueError):
            CircularROI(center=(0, 0), radius=-1)


class TestRectangularROI(unittest.TestCase):
    """Test cases for RectangularROI class."""
    
    def setUp(self):
        """Set up test cases."""
        self.rect = RectangularROI(corner1=(0, 0), corner2=(5, 5), name="test_rect")
    
    def test_initialization(self):
        """Test initialization of RectangularROI."""
        self.assertEqual(self.rect.name, "test_rect")
        np.testing.assert_array_equal(self.rect.corner1, np.array([0, 0]))
        np.testing.assert_array_equal(self.rect.corner2, np.array([5, 5]))
        np.testing.assert_array_equal(self.rect.min_coords, np.array([0, 0]))
        np.testing.assert_array_equal(self.rect.max_coords, np.array([5, 5]))
    
    def test_single_point_contains(self):
        """Test __contains__ with single points."""
        # Points inside rectangle
        self.assertTrue((0, 0) in self.rect)    # Corner
        self.assertTrue((5, 5) in self.rect)    # Corner
        self.assertTrue((2, 2) in self.rect)    # Inside
        self.assertTrue((0, 5) in self.rect)    # Edge
        self.assertTrue((5, 0) in self.rect)    # Edge
        
        # Points outside rectangle
        self.assertFalse((6, 0) in self.rect)   # Outside
        self.assertFalse((0, 6) in self.rect)   # Outside
        self.assertFalse((-1, 0) in self.rect)  # Outside
    
    def test_array_contains(self):
        """Test .contains() with arrays of points."""
        points = np.array([
            [0, 0],    # Corner
            [5, 5],    # Corner
            [2, 2],    # Inside
            [0, 5],    # Edge
            [5, 0],    # Edge
            [6, 0],    # Outside
            [0, 6],    # Outside
            [-1, 0]    # Outside
        ])
        
        expected = np.array([True, True, True, True, True, False, False, False])
        result = self.rect.contains(points)
        np.testing.assert_array_equal(result, expected)
        # __contains__ should raise ValueError for arrays
        with self.assertRaises(ValueError):
            _ = points in self.rect
    
    def test_reversed_corners(self):
        """Test rectangle with reversed corner order."""
        rect = RectangularROI(corner1=(5, 5), corner2=(0, 0))
        self.assertTrue((2, 2) in rect)
        self.assertTrue((0, 0) in rect)
        self.assertTrue((5, 5) in rect)
        self.assertFalse((6, 0) in rect)


if __name__ == '__main__':
    unittest.main() 