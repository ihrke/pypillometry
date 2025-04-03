import unittest
import sys
import numpy as np
sys.path.insert(0,"..")
import pypillometry as pp
from pypillometry import EyeDataDict

class TestEyeDataDict(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create a basic dictionary with test data
        self.test_data = {
            'left_x': np.array([1.0, 2.0, 3.0]),
            'left_y': np.array([4.0, 5.0, 6.0]),
            'left_pupil': np.array([7.0, 8.0, 9.0]),
            'right_x': np.array([10.0, 11.0, 12.0]),
            'right_y': np.array([13.0, 14.0, 15.0]),
            'right_pupil': np.array([16.0, 17.0, 18.0])
        }

    def test_initialization(self):
        """Test basic initialization of EyeDataDict"""
        # Test initialization with no arguments
        d = EyeDataDict()
        self.assertEqual(len(d), 0)
        
        # Test initialization with valid data
        d = EyeDataDict(**self.test_data)
        self.assertEqual(len(d), 3)  # length is the length of arrays, not number of keys
        self.assertEqual(len(d.data), 6)  # number of keys in data dictionary
        for key, value in self.test_data.items():
            self.assertIn(key, d)
            np.testing.assert_array_equal(d[key], value)
            self.assertEqual(d[key].dtype, np.float64)
            self.assertEqual(d[key].shape, (3,))

    def test_set_item(self):
        """Test setting items in the dictionary"""
        d = EyeDataDict()
        
        # Test setting valid data
        d["left_x"] = np.array([1.0, 2.0, 3.0])
        self.assertEqual(d["left_x"].__class__, np.ndarray)
        self.assertEqual(d["left_x"].dtype, np.float64)
        self.assertEqual(d["left_x"].shape, (3,))
        
        # Test setting data with wrong length
        with self.assertRaises(ValueError):
            d["left_y"] = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Test setting data with wrong type
        with self.assertRaises(ValueError):
            d["left_pupil"] = np.array(["a", "b", "c"])
        
        # Test setting data with wrong shape (2D array)
        with self.assertRaises(ValueError):
            d["right_x"] = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_get_item(self):
        """Test getting items from the dictionary"""
        d = EyeDataDict(**self.test_data)
        
        # Test getting existing items
        for key, value in self.test_data.items():
            np.testing.assert_array_equal(d[key], value)
        
        # Test getting non-existent items
        with self.assertRaises(KeyError):
            d["nonexistent"]

    def test_drop_empty(self):
        """Test dropping empty or None values"""
        d = EyeDataDict(**self.test_data)
        
        # Add empty and None values
        d["empty"] = np.array([])
        d["none"] = None
        
        # Verify they are dropped
        self.assertNotIn("empty", d)
        self.assertNotIn("none", d)
        
        # Verify original data is preserved
        for key, value in self.test_data.items():
            self.assertIn(key, d)
            np.testing.assert_array_equal(d[key], value)

    def test_update(self):
        """Test updating the dictionary with new values"""
        d = EyeDataDict(**self.test_data)
        
        # Update with valid data
        new_data = {
            'left_x': np.array([2.0, 3.0, 4.0]),
            'right_pupil': np.array([3.0, 4.0, 5.0])
        }
        d.update(new_data)
        
        # Verify updates
        np.testing.assert_array_equal(d['left_x'], new_data['left_x'])
        np.testing.assert_array_equal(d['right_pupil'], new_data['right_pupil'])
        
        # Verify other data is unchanged
        np.testing.assert_array_equal(d['left_y'], self.test_data['left_y'])
        np.testing.assert_array_equal(d['right_x'], self.test_data['right_x'])

    def test_get_available_eyes(self):
        """Test getting available eyes"""
        d = EyeDataDict(**self.test_data)
        eyes = d.get_available_eyes()
        self.assertEqual(set(eyes), {'left', 'right'})
        
        # Test getting eyes for specific variable
        pupil_eyes = d.get_available_eyes('pupil')
        self.assertEqual(set(pupil_eyes), {'left', 'right'})

    def test_get_available_variables(self):
        """Test getting available variables"""
        d = EyeDataDict(**self.test_data)
        variables = d.get_available_variables()
        self.assertEqual(set(variables), {'x', 'y', 'pupil'})

    def test_get_eye(self):
        """Test getting data for a specific eye"""
        d = EyeDataDict(**self.test_data)
        left_eye = d.get_eye('left')
        self.assertEqual(len(left_eye.data), 3)  # x, y, pupil
        self.assertIn('left_x', left_eye)
        self.assertIn('left_y', left_eye)
        self.assertIn('left_pupil', left_eye)

    def test_get_variable(self):
        """Test getting data for a specific variable"""
        d = EyeDataDict(**self.test_data)
        pupil_data = d.get_variable('pupil')
        self.assertEqual(len(pupil_data.data), 2)  # left and right
        self.assertIn('left_pupil', pupil_data)
        self.assertIn('right_pupil', pupil_data)

if __name__ == '__main__':
    unittest.main()