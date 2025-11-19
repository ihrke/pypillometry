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

    def test_set_with_mask_explicit(self):
        """Test set_with_mask with explicit mask provided"""
        d = EyeDataDict()
        
        # Create data and explicit mask
        data = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([0, 1, 0, 1])
        
        # Set with explicit mask
        d.set_with_mask('left_pupil', data, mask=mask)
        
        # Verify data
        np.testing.assert_array_equal(d['left_pupil'], data)
        # Verify mask
        np.testing.assert_array_equal(d.mask['left_pupil'], mask)
    
    def test_set_with_mask_preserve_existing(self):
        """Test set_with_mask with preserve_mask=True"""
        d = EyeDataDict()
        
        # Initial data and mask
        initial_data = np.array([1.0, 2.0, 3.0, 4.0])
        initial_mask = np.array([0, 1, 0, 1])
        d['left_pupil'] = initial_data
        d.mask['left_pupil'] = initial_mask
        
        # Update data while preserving mask
        new_data = np.array([10.0, 20.0, 30.0, 40.0])
        d.set_with_mask('left_pupil', new_data, preserve_mask=True)
        
        # Verify new data
        np.testing.assert_array_equal(d['left_pupil'], new_data)
        # Verify mask is preserved
        np.testing.assert_array_equal(d.mask['left_pupil'], initial_mask)
    
    def test_set_with_mask_default_behavior(self):
        """Test set_with_mask with default behavior (same as __setitem__)"""
        d = EyeDataDict()
        
        # Initial data and mask
        initial_data = np.array([1.0, 2.0, 3.0, 4.0])
        initial_mask = np.array([0, 1, 0, 1])
        d['left_pupil'] = initial_data
        d.mask['left_pupil'] = initial_mask
        
        # Update data with default behavior (resets mask)
        new_data = np.array([10.0, 20.0, 30.0, 40.0])
        d.set_with_mask('left_pupil', new_data)
        
        # Verify new data
        np.testing.assert_array_equal(d['left_pupil'], new_data)
        # Verify mask is reset to zeros
        np.testing.assert_array_equal(d.mask['left_pupil'], np.zeros(4, dtype=int))
    
    def test_set_with_mask_with_nans(self):
        """Test set_with_mask handles NaN values correctly"""
        d = EyeDataDict()
        
        # Data with NaN values
        data = np.array([1.0, np.nan, 3.0, np.nan])
        d.set_with_mask('left_pupil', data)
        
        # Verify NaN positions are masked
        expected_mask = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(d.mask['left_pupil'], expected_mask)
    
    def test_set_with_mask_with_tuple_key(self):
        """Test set_with_mask works with tuple keys"""
        d = EyeDataDict()
        
        # Set with tuple key
        data = np.array([1.0, 2.0, 3.0])
        mask = np.array([0, 1, 0])
        d.set_with_mask(('left', 'pupil'), data, mask=mask)
        
        # Verify data and mask
        np.testing.assert_array_equal(d['left_pupil'], data)
        np.testing.assert_array_equal(d.mask['left_pupil'], mask)
    
    def test_set_with_mask_shape_validation(self):
        """Test set_with_mask validates array shapes"""
        d = EyeDataDict()
        
        # Set initial data
        d['left_x'] = np.array([1.0, 2.0, 3.0])
        
        # Try to set data with wrong shape
        with self.assertRaises(ValueError):
            d.set_with_mask('left_y', np.array([1.0, 2.0, 3.0, 4.0]))
    
    def test_set_with_mask_preserve_on_nonexistent_key(self):
        """Test set_with_mask with preserve_mask on a new key creates zero mask"""
        d = EyeDataDict()
        
        # Set new key with preserve_mask=True (but no existing mask to preserve)
        data = np.array([1.0, 2.0, 3.0])
        d.set_with_mask('left_pupil', data, preserve_mask=True)
        
        # Verify data
        np.testing.assert_array_equal(d['left_pupil'], data)
        # Verify mask is created as zeros (since there was no mask to preserve)
        np.testing.assert_array_equal(d.mask['left_pupil'], np.zeros(3, dtype=int))

    def test_count_masked(self):
        """Test counting missing values from masks"""
        d = EyeDataDict()
        
        # Create test data with masks
        d['left_x'] = np.array([1.0, 2.0, 3.0, 4.0])
        d['left_y'] = np.array([5.0, 6.0, 7.0, 8.0])
        d['right_x'] = np.array([9.0, 10.0, 11.0, 12.0])
        
        # Set masks with different patterns of missing values
        d.set_mask('left_x', np.array([0, 1, 0, 1]))  # 2 missing values
        d.set_mask('left_y', np.array([1, 0, 1, 0]))  # 2 missing values
        d.set_mask('right_x', np.array([0, 0, 1, 0]))  # 1 missing value
        
        # Test total count (max across keys)
        self.assertEqual(d.count_masked(), 2)  # max of [2, 2, 1]
        
        # Test per-key counts
        per_key_counts = d.count_masked(per_key=True)
        self.assertEqual(per_key_counts['left_x'], 2)
        self.assertEqual(per_key_counts['left_y'], 2)
        self.assertEqual(per_key_counts['right_x'], 1)
        
        # Test with no missing values
        d.set_mask('left_x', np.zeros(4, dtype=int))
        self.assertEqual(d.count_masked(), 2)  # max of [0, 2, 1]
        per_key_counts = d.count_masked(per_key=True)
        self.assertEqual(per_key_counts['left_x'], 0)
        
        # Test with all values missing in one key
        d.set_mask('left_y', np.ones(4, dtype=int))
        self.assertEqual(d.count_masked(), 4)  # max of [0, 4, 1]
        per_key_counts = d.count_masked(per_key=True)
        self.assertEqual(per_key_counts['left_y'], 4)

    def test_get_mask(self):
        """Test getting masks from EyeDataDict"""
        d = EyeDataDict()
        
        # Create test data with masks
        d['left_x'] = np.array([1.0, 2.0, 3.0, 4.0])
        d['left_y'] = np.array([5.0, 6.0, 7.0, 8.0])
        d['right_x'] = np.array([9.0, 10.0, 11.0, 12.0])
        
        # Set masks with different patterns
        d.set_mask('left_x', np.array([0, 1, 0, 1]))  # second and fourth values masked
        d.set_mask('left_y', np.array([1, 0, 1, 0]))  # first and third values masked
        d.set_mask('right_x', np.array([0, 0, 1, 0]))  # third value masked
        
        # Test getting specific masks
        np.testing.assert_array_equal(d.get_mask('left_x'), np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(d.get_mask('left_y'), np.array([1, 0, 1, 0]))
        np.testing.assert_array_equal(d.get_mask('right_x'), np.array([0, 0, 1, 0]))
        
        # Test getting joint mask (logical OR of all masks)
        joint_mask = d.get_mask()
        expected_joint = np.array([1, 1, 1, 1])  # Any value masked in any key is masked in joint
        np.testing.assert_array_equal(joint_mask, expected_joint)
        
        # Test with invalid key
        with self.assertRaises(KeyError):
            d.get_mask('nonexistent')
            
        # Test with no masks set (should return zeros)
        d2 = EyeDataDict()
        d2['left_pupil'] = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(d2.get_mask('left_pupil'), np.zeros(3, dtype=int))
        np.testing.assert_array_equal(d2.get_mask(), np.zeros(3, dtype=int))

if __name__ == '__main__':
    unittest.main()