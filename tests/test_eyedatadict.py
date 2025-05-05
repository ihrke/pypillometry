import unittest
import sys
import numpy as np
import tempfile
import os
import shutil
sys.path.insert(0,"..")
import pypillometry as pp
from pypillometry import EyeDataDict, CachedEyeDataDict

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
        # Create a temporary directory for cache tests
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary cache directory
        shutil.rmtree(self.cache_dir)

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

    def test_cached_initialization(self):
        """Test initialization of CachedEyeDataDict"""
        # Test initialization with memory cache only
        d = CachedEyeDataDict(max_memory_mb=1)
        self.assertEqual(len(d), 0)
        self.assertTrue(hasattr(d, '_in_memory_data'))
        self.assertTrue(hasattr(d, '_h5_file'))
        
        # Test initialization with disk cache
        d = CachedEyeDataDict(cache_dir=self.cache_dir, max_memory_mb=1)
        self.assertEqual(len(d), 0)
        self.assertEqual(d._cache_dir, self.cache_dir)
        self.assertTrue(os.path.exists(os.path.join(self.cache_dir, 'eyedata_cache.h5')))

    def test_cached_set_get(self):
        """Test setting and getting items with caching"""
        # Initialize with empty dictionary
        d = CachedEyeDataDict(cache_dir=self.cache_dir, max_memory_mb=1)
        
        # Create a test array
        data = np.random.rand(1000)
        
        # Set data
        d['test_data'] = data
        
        # Verify data is accessible
        np.testing.assert_array_equal(d['test_data'], data)
        
        # Verify data is in HDF5 file
        self.assertTrue('test_data' in d._h5_file['data'])

    def test_cached_lru_eviction(self):
        """Test LRU eviction of cached items"""
        # Initialize with very small memory limit
        d = CachedEyeDataDict(cache_dir=self.cache_dir, max_memory_mb=0.1)
        
        # Add multiple arrays of the same size
        arrays = {}
        for i in range(5):
            key = f'array_{i}'
            arrays[key] = np.random.rand(1000)  # ~8KB each
            d[key] = arrays[key]
        
        # Access items in specific order to test LRU
        d['array_0']  # Most recently used
        d['array_2']  # Second most recently used
        
        # Add new data to force eviction
        d['new_array'] = np.random.rand(1000)
        
        # Verify most recently accessed items are still in memory
        self.assertIn('array_0', d._in_memory_data)
        self.assertIn('array_2', d._in_memory_data)
        
        # Verify other items are in HDF5 file
        for i in range(5):
            key = f'array_{i}'
            if key not in d._in_memory_data:
                self.assertTrue(key in d._h5_file['data'])
                # Verify data integrity
                np.testing.assert_array_equal(d[key], arrays[key])

    def test_cached_update(self):
        """Test updating cached dictionary"""
        d = CachedEyeDataDict(cache_dir=self.cache_dir, max_memory_mb=1)
        
        # Initial data
        d.update(self.test_data)
        
        # Verify all data is accessible
        for key, value in self.test_data.items():
            np.testing.assert_array_equal(d[key], value)
        
        # Update with new data of the same shape
        new_data = {
            'left_x': np.array([10.0, 20.0, 30.0]),
            'new_key': np.array([1.0, 2.0, 3.0])
        }
        d.update(new_data)
        
        # Verify updates
        np.testing.assert_array_equal(d['left_x'], new_data['left_x'])
        np.testing.assert_array_equal(d['new_key'], new_data['new_key'])
        
        # Verify other data is unchanged
        for key in ['left_y', 'right_x', 'right_y']:
            np.testing.assert_array_equal(d[key], self.test_data[key])

    def test_cached_clear(self):
        """Test clearing cached dictionary"""
        d = CachedEyeDataDict(cache_dir=self.cache_dir, max_memory_mb=1)
        
        # Add data
        d.update(self.test_data)
        d['new_data'] = np.array([4.0, 5.0, 6.0])  # Same shape as test_data
        
        # Clear dictionary
        d.clear_cache()
        
        # Verify memory cache is empty
        self.assertEqual(len(d._in_memory_data), 0)
        
        # Verify HDF5 file is empty
        self.assertEqual(len(d._h5_file['data']), 0)
        self.assertEqual(len(d._h5_file['mask']), 0)
        
        # Verify dictionary is empty
        self.assertEqual(len(d), 0)

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

    def test_cached_get_mask(self):
        """Test getting masks from CachedEyeDataDict"""
        d = CachedEyeDataDict(cache_dir=self.cache_dir, max_memory_mb=1)
        
        # Create test data with masks
        d['left_x'] = np.array([1.0, 2.0, 3.0, 4.0])
        d['left_y'] = np.array([5.0, 6.0, 7.0, 8.0])
        d['right_x'] = np.array([9.0, 10.0, 11.0, 12.0])
        
        # Set masks with different patterns
        d.set_mask('left_x', np.array([0, 1, 0, 1]))  # second and fourth values masked
        d.set_mask('left_y', np.array([1, 0, 1, 0]))  # first and third values masked
        d.set_mask('right_x', np.array([0, 0, 1, 0]))  # third value masked
        
        # Test getting specific masks (from memory cache)
        np.testing.assert_array_equal(d.get_mask('left_x'), np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(d.get_mask('left_y'), np.array([1, 0, 1, 0]))
        np.testing.assert_array_equal(d.get_mask('right_x'), np.array([0, 0, 1, 0]))
        
        # Force data to be written to HDF5
        d._h5_file.flush()
        
        # Clear memory cache to test HDF5 loading
        d._in_memory_data.clear()
        d._in_memory_mask.clear()
        d._current_memory_bytes = 0
        
        # Test getting masks from HDF5
        np.testing.assert_array_equal(d.get_mask('left_x'), np.array([0, 1, 0, 1]))
        np.testing.assert_array_equal(d.get_mask('left_y'), np.array([1, 0, 1, 0]))
        np.testing.assert_array_equal(d.get_mask('right_x'), np.array([0, 0, 1, 0]))
        
        # Test getting joint mask (should work with both memory and HDF5 data)
        joint_mask = d.get_mask()
        expected_joint = np.array([1, 1, 1, 1])  # Any value masked in any key is masked in joint
        np.testing.assert_array_equal(joint_mask, expected_joint)
        
        # Test with invalid key
        with self.assertRaises(KeyError):
            d.get_mask('left_invalid')  # Valid format but non-existent key
            
        # Test with no masks set (should return zeros)
        d2 = CachedEyeDataDict(cache_dir=self.cache_dir, max_memory_mb=1)
        d2['left_pupil'] = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(d2.get_mask('left_pupil'), np.zeros(3, dtype=int))
        np.testing.assert_array_equal(d2.get_mask(), np.zeros(3, dtype=int))

if __name__ == '__main__':
    unittest.main()