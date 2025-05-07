import unittest
import sys
import numpy as np

from pypillometry.eyedata.eyedatadict import EyeDataDict
from pypillometry.eyedata.generic import GenericEyeData
sys.path.insert(0,"..")
import pypillometry as pp
from pypillometry.example_data import get_rlmw_002_short
from pypillometry.plot import EyePlotter
import pytest

class TestEyeData(unittest.TestCase):
    def setUp(self):
        """Set up test data using the example dataset"""
        # Get example data
        self.eyedata = get_rlmw_002_short()
        
        # Store some key attributes for testing
        self.time = self.eyedata.tx
        self.left_x = self.eyedata.data['left_x']
        self.left_y = self.eyedata.data['left_y']
        self.left_pupil = self.eyedata.data['left_pupil']
        self.right_x = self.eyedata.data['right_x']
        self.right_y = self.eyedata.data['right_y']
        self.right_pupil = self.eyedata.data['right_pupil']
        self.event_onsets = self.eyedata.event_onsets
        self.event_labels = self.eyedata.event_labels

    def test_initialization(self):
        """Test basic initialization of EyeData"""
        # Test with minimal required data
        d = pp.EyeData(left_x=[1,2], left_y=[3,4], left_pupil=[5,6], sampling_rate=1000)
        self.assertEqual(d.__class__, pp.EyeData)
        self.assertIn('left_pupil', d.data)
        np.testing.assert_array_equal(d.data['left_pupil'], [5,6])
        
        # Test with invalid data (missing y coordinate)
        with self.assertRaises(ValueError):
            pp.EyeData(left_x=[1,2])
            
        # Test with complete data
        self.assertEqual(self.eyedata.name, "test short")
        self.assertEqual(self.eyedata.fs, 500.0)  # Updated to match actual sampling rate
        self.assertEqual(self.eyedata.screen_width, 1280)  # Updated to match actual screen resolution
        self.assertEqual(self.eyedata.screen_height, 1024)
        self.assertEqual(self.eyedata.physical_screen_width, 30.0)  # Updated to match actual physical size
        self.assertEqual(self.eyedata.physical_screen_height, 20.0)
        self.assertEqual(self.eyedata.screen_eye_distance, 60.0)

        # Test pupil data initialization
        # Check that pupil data exists for both eyes
        self.assertIn('left_pupil', self.eyedata.data)
        self.assertIn('right_pupil', self.eyedata.data)
        
        # Check pupil data shape matches time array
        self.assertEqual(len(self.eyedata.data['left_pupil']), len(self.eyedata.tx))
        self.assertEqual(len(self.eyedata.data['right_pupil']), len(self.eyedata.tx))
        
        # Check pupil data contains valid values (not NaN or infinite)
        self.assertTrue(np.all(np.isfinite(self.eyedata.data['left_pupil'])))
        self.assertTrue(np.all(np.isfinite(self.eyedata.data['right_pupil'])))
        
        # Check pupil data is non-negative
        self.assertTrue(np.all(self.eyedata.data['left_pupil'] >= 0))
        self.assertTrue(np.all(self.eyedata.data['right_pupil'] >= 0))

    def test_initialization_combinations(self):
        """Test initialization with different combinations of eye data"""
        # Test with only left eye data (x, y, pupil)
        d_left = pp.EyeData(left_x=[1,2], left_y=[3,4], left_pupil=[5,6], sampling_rate=1000)
        self.assertIn('left_x', d_left.data)
        self.assertIn('left_y', d_left.data)
        self.assertIn('left_pupil', d_left.data)
        self.assertNotIn('right_x', d_left.data)
        self.assertNotIn('right_y', d_left.data)
        self.assertNotIn('right_pupil', d_left.data)
        
        # Test with only right eye data (x, y, pupil)
        d_right = pp.EyeData(right_x=[1,2], right_y=[3,4], right_pupil=[5,6], sampling_rate=1000)
        self.assertNotIn('left_x', d_right.data)
        self.assertNotIn('left_y', d_right.data)
        self.assertNotIn('left_pupil', d_right.data)
        self.assertIn('right_x', d_right.data)
        self.assertIn('right_y', d_right.data)
        self.assertIn('right_pupil', d_right.data)
        
        # Test with both eyes (x, y, pupil)
        d_both = pp.EyeData(left_x=[1,2], left_y=[3,4], left_pupil=[5,6],
                           right_x=[7,8], right_y=[9,10], right_pupil=[11,12],
                           sampling_rate=1000)
        self.assertIn('left_x', d_both.data)
        self.assertIn('left_y', d_both.data)
        self.assertIn('left_pupil', d_both.data)
        self.assertIn('right_x', d_both.data)
        self.assertIn('right_y', d_both.data)
        self.assertIn('right_pupil', d_both.data)
        
        # Test with left eye (x, y) but no pupil
        d_left_no_pupil = pp.EyeData(left_x=[1,2], left_y=[3,4], sampling_rate=1000)
        self.assertIn('left_x', d_left_no_pupil.data)
        self.assertIn('left_y', d_left_no_pupil.data)
        self.assertNotIn('left_pupil', d_left_no_pupil.data)
        
        # Test with right eye (x, y) but no pupil
        d_right_no_pupil = pp.EyeData(right_x=[1,2], right_y=[3,4], sampling_rate=1000)
        self.assertIn('right_x', d_right_no_pupil.data)
        self.assertIn('right_y', d_right_no_pupil.data)
        self.assertNotIn('right_pupil', d_right_no_pupil.data)
        
        # Test with left eye (x, pupil) but no y - should raise ValueError
        with self.assertRaises(ValueError):
            pp.EyeData(left_x=[1,2], left_pupil=[5,6], sampling_rate=1000)
            
        # Test with right eye (y, pupil) but no x - should raise ValueError
        with self.assertRaises(ValueError):
            pp.EyeData(right_y=[3,4], right_pupil=[5,6], sampling_rate=1000)
            
        # Test with no eye data at all - should raise ValueError
        with self.assertRaises(ValueError):
            pp.EyeData(sampling_rate=1000)
            
        # Test with left eye (x) but no y - should raise ValueError
        with self.assertRaises(ValueError):
            pp.EyeData(left_x=[1,2], sampling_rate=1000)
            
        # Test with right eye (y) but no x - should raise ValueError
        with self.assertRaises(ValueError):
            pp.EyeData(right_y=[3,4], sampling_rate=1000)

    def test_summary(self):
        """Test the summary method"""
        summary = self.eyedata.summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('sampling_rate', summary)
        self.assertIn('duration_minutes', summary)  # Updated to match actual key
        self.assertIn('nevents', summary)  # Updated to match actual key

    def test_get_pupildata(self):
        """Test the get_pupildata method"""
        # Test getting left eye pupil data
        left_pd = self.eyedata.get_pupildata(eye='left')
        self.assertEqual(left_pd.__class__, pp.PupilData)
        np.testing.assert_array_equal(left_pd.tx, self.time)
        np.testing.assert_array_equal(left_pd.data['left_pupil'], self.left_pupil)
        
        # Test getting right eye pupil data
        right_pd = self.eyedata.get_pupildata(eye='right')
        self.assertEqual(right_pd.__class__, pp.PupilData)
        np.testing.assert_array_equal(right_pd.tx, self.time)
        np.testing.assert_array_equal(right_pd.data['right_pupil'], self.right_pupil)
        
        # Test with no eye specified when multiple eyes are present
        with self.assertRaises(ValueError):
            self.eyedata.get_pupildata()
            
        # Test with invalid eye
        with self.assertRaises(ValueError):
            self.eyedata.get_pupildata(eye='invalid')

    def test_correct_pupil_foreshortening(self):
        """Test the correct_pupil_foreshortening method"""
        # Test correction for both eyes

        corrected = self.eyedata.correct_pupil_foreshortening(eyes=['left', 'right'])
        self.assertEqual(corrected.__class__, pp.EyeData)
        self.assertEqual(corrected.data['left_pupil'].shape, self.left_pupil.shape)
        # Test correction for single eye
        corrected_left = self.eyedata.correct_pupil_foreshortening(eyes=['left'])
        self.assertEqual(corrected_left.__class__, pp.EyeData)
        
        # Test with custom midpoint
        corrected_custom = self.eyedata.correct_pupil_foreshortening(
            eyes=['left'], 
            midpoint=(640, 512)  # Center of 1280x1024 screen
        )
        self.assertEqual(corrected_custom.__class__, pp.EyeData)
        
        # Test inplace modification
        original_pupil = self.eyedata.data['left_pupil'].copy()
        self.eyedata.correct_pupil_foreshortening(eyes=['left'], inplace=True)
        self.assertFalse(np.array_equal(original_pupil, self.eyedata.data['left_pupil']))

    def test_plot_property(self):
        """Test the plot property"""
        plotter = self.eyedata.plot
        self.assertEqual(plotter.__class__, EyePlotter)
        self.assertEqual(plotter.obj, self.eyedata)  # Updated to use obj instead of data

    def test_scale_and_unscale(self):
        """Test scaling and unscaling of data"""
        # Create artificial data with known mean and std
        original_data = np.random.normal(5.0, 2.0, len(self.eyedata))
        self.eyedata.data['left_pupil'] = original_data.copy()  # Make a copy to ensure no modifications
        
        # Test scaling with default parameters (using data's mean and std)
        scaled = self.eyedata.scale(variables=['pupil'], eyes=['left'])
        self.assertNotEqual(np.mean(scaled.data['left_pupil']), np.mean(original_data))
        self.assertAlmostEqual(np.mean(scaled.data['left_pupil']), 0.0, places=6)
        self.assertAlmostEqual(np.std(scaled.data['left_pupil']), 1.0, places=6)
        
        # Test unscaling
        unscaled = scaled.unscale(variables=['pupil'], eyes=['left'])
        np.testing.assert_array_almost_equal(unscaled.data['left_pupil'], original_data)
        
        # Test that original data is unchanged
        np.testing.assert_array_almost_equal(self.eyedata.data['left_pupil'], original_data)

    def test_downsample(self):
        """Test downsampling of data"""
        # Test downsampling to half the original sampling rate
        original_fs = self.eyedata.fs
        original_len = len(self.eyedata)
        downsampled = self.eyedata.downsample(fsd=original_fs/2)
        self.assertEqual(downsampled.fs, original_fs/2)
        # Allow for small differences in length due to rounding
        self.assertAlmostEqual(len(downsampled), original_len/2, delta=1)
        
        # Test downsampling with decimate factor
        # When dsfac=True, fsd is the decimate factor itself
        # Use a fresh object for this test
        fresh_eyedata = get_rlmw_002_short()
        original_fs = fresh_eyedata.fs
        original_len = len(fresh_eyedata)
        dsfac = 4  # Use a factor of 4 to get 125 Hz
        downsampled_factor = fresh_eyedata.downsample(fsd=dsfac, dsfac=True)
        self.assertEqual(downsampled_factor.fs, 125.0)  # fs should be 125 Hz (500/4)
        self.assertAlmostEqual(len(downsampled_factor), original_len/dsfac, delta=1)  # length should be original_len/dsfac

    def test_merge_eyes(self):
        """Test merging data from both eyes"""
        # Test merging with mean method
        merged = self.eyedata.merge_eyes(eyes=['left', 'right'], variables=['pupil'])
        self.assertIn('mean_pupil', merged.data)
        np.testing.assert_array_almost_equal(
            merged.data['mean_pupil'],
            (self.eyedata.data['left_pupil'] + self.eyedata.data['right_pupil']) / 2
        )
        
        # Test merging without keeping original eyes
        merged_no_keep = self.eyedata.merge_eyes(eyes=['left', 'right'], 
                                                variables=['pupil'], keep_eyes=False)
        self.assertNotIn('left_pupil', merged_no_keep.data)
        self.assertNotIn('right_pupil', merged_no_keep.data)
        self.assertIn('mean_pupil', merged_no_keep.data)

    def test_blinks_merge(self):
        """Test merging of close blinks"""
        # First create some artificial blinks
        self.eyedata.set_blinks('left', 'pupil', np.array([[100, 200], [250, 350]]))
        
        # Merge blinks that are within 100ms of each other
        merged = self.eyedata.blinks_merge(eyes=['left'], variables=['pupil'], distance=100)
        blinks = merged.get_blinks('left', 'pupil')
        self.assertEqual(len(blinks), 1)  # Should merge into one blink
        self.assertEqual(blinks[0][0], 100)  # Start of first blink
        self.assertEqual(blinks[0][1], 350)  # End of last blink

    def test_stat_per_event(self):
        """Test statistical analysis per event"""
        # Create some test events
        self.eyedata.event_onsets = np.array([1000, 2000, 3000])
        self.eyedata.event_labels = np.array(['stim1', 'stim2', 'stim3'])
        
        # Test mean pupil size in interval around events
        stats = self.eyedata.stat_per_event(
            interval=(-100, 100),
            event_select='stim',
            eyes=['left'],
            variables=['pupil'],
            statfct=np.mean
        )
        self.assertIn('left_pupil', stats)
        self.assertEqual(len(stats['left_pupil']), 3)  # One value per event

    def test_get_duration(self):
        """Test getting duration in different units"""
        duration_min = self.eyedata.get_duration(units='min')
        duration_sec = self.eyedata.get_duration(units='sec')
        duration_h = self.eyedata.get_duration(units='h')
        
        # Check relationships between different units
        self.assertAlmostEqual(duration_min * 60, duration_sec)
        self.assertAlmostEqual(duration_h * 60, duration_min)

    def test_get_size(self):
        """Test size reporting."""
        size = self.eyedata.get_size()
        self.assertIsInstance(size, (int, dict))

    def test_write_and_read_file(self):
        """Test writing and reading dataset to/from file"""
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            # Write dataset
            self.eyedata.write_file(tmp.name)
            
            # Read dataset back
            loaded = pp.EyeData.from_file(tmp.name)
            
            # Compare key attributes
            np.testing.assert_array_equal(loaded.tx, self.eyedata.tx)
            np.testing.assert_array_equal(loaded.data['left_pupil'], self.eyedata.data['left_pupil'])
            self.assertEqual(loaded.fs, self.eyedata.fs)
            
            # Clean up
            os.unlink(tmp.name)

    def test_merge_masks(self):
        """Test merging masks across all variables."""
        # Create test data with different masks
        data = {
            'left_pupil': np.array([1, 2, 3, 4, 5]),
            'right_pupil': np.array([1, 2, 3, 4, 5]),
            'left_x': np.array([1, 2, 3, 4, 5]),
            'left_y': np.array([1, 2, 3, 4, 5]),
            'right_x': np.array([1, 2, 3, 4, 5]),
            'right_y': np.array([1, 2, 3, 4, 5])
        }
        masks = {
            'left_pupil': np.array([0, 1, 0, 1, 0]),
            'right_pupil': np.array([1, 0, 1, 0, 1]),
            'left_x': np.array([0, 0, 1, 1, 0]),
            'left_y': np.array([0, 0, 1, 1, 0]),
            'right_x': np.array([0, 0, 1, 1, 0]),
            'right_y': np.array([0, 0, 1, 1, 0])
        }
        
        # Create EyeData object
        obj = pp.EyeData(
            left_x=data['left_x'],
            left_y=data['left_y'],
            left_pupil=data['left_pupil'],
            right_x=data['right_x'],
            right_y=data['right_y'],
            right_pupil=data['right_pupil'],
            sampling_rate=1000
        )
        
        # Set masks directly on the EyeData object
        for k, v in masks.items():
            obj.data.set_mask(k, v)
        
        # Test in-place merging
        obj.merge_masks(inplace=True)
        
        # Expected joint mask (logical OR of all masks)
        expected_mask = np.array([1, 1, 1, 1, 1])
        
        # Check that all variables have the joint mask
        for key in data.keys():
            np.testing.assert_array_equal(obj.data.mask[key], expected_mask)
            
        # Test copy behavior
        obj2 = obj.merge_masks(inplace=False)
        self.assertIsNot(obj, obj2)  # Should be different objects
        
        # Check that original object still has joint mask
        for key in data.keys():
            np.testing.assert_array_equal(obj.data.mask[key], expected_mask)
            
        # Check that copy also has joint mask
        for key in data.keys():
            np.testing.assert_array_equal(obj2.data.mask[key], expected_mask)
            
        # Test with no masks
        obj3 = pp.EyeData(
            left_x=data['left_x'],
            left_y=data['left_y'],
            left_pupil=data['left_pupil'],
            right_x=data['right_x'],
            right_y=data['right_y'],
            right_pupil=data['right_pupil'],
            sampling_rate=1000
        )
        
        # Should not raise any errors
        obj3.merge_masks()
        
        # Check that all masks are empty (all zeros)
        expected_empty_mask = np.zeros(len(data['left_pupil']))
        for key in data.keys():
            np.testing.assert_array_equal(obj3.data.mask[key], expected_empty_mask)

if __name__ == '__main__':
    unittest.main() 