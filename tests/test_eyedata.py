import unittest
import sys
import numpy as np

from pypillometry.eyedata.eyedatadict import EyeDataDict
from pypillometry.eyedata.generic import GenericEyeData
from pypillometry.eyedata import ExperimentalSetup
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
        self.assertEqual(self.eyedata.experimental_setup.screen_width, 1280)  # Updated to match actual screen resolution
        self.assertEqual(self.eyedata.experimental_setup.screen_height, 1024)
        self.assertEqual(self.eyedata.experimental_setup.physical_screen_width, 300.0)  # 30 cm = 300 mm
        self.assertEqual(self.eyedata.experimental_setup.physical_screen_height, 200.0)  # 20 cm = 200 mm
        self.assertEqual(self.eyedata.experimental_setup.d, 600.0)  # 60 cm = 600 mm
    
    def test_camera_position_initialization(self):
        """Test initialization with camera position via ExperimentalSetup"""
        # Test with camera_spherical (theta, phi, r) in eye frame
        setup = ExperimentalSetup(
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            eye_to_screen_center="700 mm"
        )
        d = pp.EyeData(
            left_x=[1,2,3], 
            left_y=[3,4,5], 
            sampling_rate=1000,
            experimental_setup=setup
        )
        self.assertEqual(d.experimental_setup.r, 600.0)
        self.assertAlmostEqual(np.degrees(d.experimental_setup.theta), 20.0, places=5)
        
        # Test that it appears in summary
        summary = d.summary()
        self.assertIn('experimental_setup', summary)
        self.assertEqual(summary['experimental_setup']['camera_distance_mm'], 600.0)
    
    def test_experimental_setup_via_set_experimental_setup(self):
        """Test setting experimental setup via set_experimental_setup"""
        d = pp.EyeData(left_x=[1,2,3], left_y=[3,4,5], sampling_rate=1000)
        
        # Should be None when not set
        self.assertIsNone(d.experimental_setup)
        
        # Set via set_experimental_setup (eye frame for direct r)
        d.set_experimental_setup(
            camera_spherical=("20 deg", "-90 deg", "550 mm"),
            eye_to_screen_center="700 mm"
        )
        self.assertAlmostEqual(d.experimental_setup.r, 550.0, places=5)
        
        # Should appear in summary
        summary = d.summary()
        self.assertAlmostEqual(summary['experimental_setup']['camera_distance_mm'], 550.0, places=5)
    
    def test_ipd_initialization(self):
        """Test initialization with inter-pupillary distance via ExperimentalSetup"""
        # Test with ipd during initialization
        setup = ExperimentalSetup(ipd="65 mm")
        d = pp.EyeData(
            left_x=[1,2,3], 
            left_y=[3,4,5], 
            right_x=[4,5,6],
            right_y=[7,8,9],
            sampling_rate=1000,
            experimental_setup=setup
        )
        self.assertEqual(d.experimental_setup.ipd, 65.0)
        
        # Test that it appears in summary
        summary = d.summary()
        self.assertIn('experimental_setup', summary)
        self.assertEqual(summary['experimental_setup']['ipd_mm'], 65.0)
    
    def test_ipd_via_set_experimental_setup(self):
        """Test setting ipd via set_experimental_setup"""
        d = pp.EyeData(left_x=[1,2,3], left_y=[3,4,5], sampling_rate=1000)
        
        # experimental_setup should be None initially
        self.assertIsNone(d.experimental_setup)
        
        # Set via set_experimental_setup
        d.set_experimental_setup(ipd="63 mm")
        self.assertEqual(d.experimental_setup.ipd, 63.0)
        
        # Should appear in summary
        summary = d.summary()
        self.assertEqual(summary['experimental_setup']['ipd_mm'], 63.0)
    
    def test_all_distance_parameters_together(self):
        """Test setting all distance parameters together via ExperimentalSetup"""
        setup = ExperimentalSetup(
            screen_resolution=(1920, 1080),
            physical_screen_size=("50 mm", "30 mm"),
            eye_to_screen_center="70 mm",
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            ipd="65 mm"
        )
        d = pp.EyeData(
            left_x=[1,2,3], 
            left_y=[3,4,5], 
            right_x=[4,5,6],
            right_y=[7,8,9],
            sampling_rate=1000,
            experimental_setup=setup
        )
        
        # Check all parameters are set
        self.assertEqual(d.experimental_setup.d, 70.0)
        self.assertEqual(d.experimental_setup.r, 600.0)
        self.assertEqual(d.experimental_setup.ipd, 65.0)
        self.assertEqual(d.experimental_setup.physical_screen_width, 50.0)
        self.assertEqual(d.experimental_setup.physical_screen_height, 30.0)
        
        # Check all appear in summary
        summary = d.summary()
        setup_summary = summary['experimental_setup']
        self.assertEqual(setup_summary['eye_to_screen_distance_mm'], 70.0)
        self.assertEqual(setup_summary['camera_distance_mm'], 600.0)
        self.assertEqual(setup_summary['ipd_mm'], 65.0)
    
    def test_distance_parameters_not_set_in_summary(self):
        """Test that unset distance parameters show as 'not set' in summary"""
        d = pp.EyeData(left_x=[1,2,3], left_y=[3,4,5], sampling_rate=1000)
        
        summary = d.summary()
        self.assertEqual(summary['experimental_setup'], 'not set')

        # Test pupil data initialization
        # Check that pupil data exists for both eyes
        self.assertIn('left_pupil', self.eyedata.data)
        self.assertIn('right_pupil', self.eyedata.data)
        
        # Check pupil data shape matches time array
        self.assertEqual(len(self.eyedata.data['left_pupil']), len(self.eyedata.tx))
        self.assertEqual(len(self.eyedata.data['right_pupil']), len(self.eyedata.tx))
        
        # Check pupil data contains valid values (finite or NaN for missing/zero values)
        # Note: zeros are automatically converted to NaN as they represent invalid pupil sizes
        left_finite = self.eyedata.data['left_pupil'][np.isfinite(self.eyedata.data['left_pupil'])]
        right_finite = self.eyedata.data['right_pupil'][np.isfinite(self.eyedata.data['right_pupil'])]
        self.assertTrue(len(left_finite) > 0)  # At least some valid data
        self.assertTrue(len(right_finite) > 0)
        
        # Check finite pupil data is positive (no zeros, as they're converted to NaN)
        self.assertTrue(np.all(left_finite > 0))
        self.assertTrue(np.all(right_finite > 0))

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

    def test_dunder_getitem_returns_masked_array(self):
        """GenericEyeData should return masked arrays via __getitem__."""
        masked = self.eyedata['left_pupil']
        self.assertIsInstance(masked, np.ma.MaskedArray)
        np.testing.assert_array_equal(np.ma.getdata(masked), self.left_pupil)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(masked),
            self.eyedata.data.mask['left_pupil']
        )

        masked_tuple = self.eyedata['left', 'pupil']
        self.assertIsInstance(masked_tuple, np.ma.MaskedArray)
        np.testing.assert_array_equal(np.ma.getdata(masked_tuple), self.left_pupil)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(masked_tuple),
            self.eyedata.data.mask['left_pupil']
        )

    def test_dunder_getitem_time_accessors(self):
        """Special time keys should return the time axis in requested units."""
        np.testing.assert_array_equal(self.eyedata['time'], self.eyedata.tx)
        np.testing.assert_allclose(self.eyedata['time_sec'], self.eyedata.tx / 1000.0)
        np.testing.assert_allclose(self.eyedata['time_min'], self.eyedata.tx / (1000.0 * 60.0))
        np.testing.assert_allclose(self.eyedata['time', 'sec'], self.eyedata.tx / 1000.0)
        
        # Test with aliases
        np.testing.assert_allclose(self.eyedata['time_seconds'], self.eyedata.tx / 1000.0)
        np.testing.assert_allclose(self.eyedata['time_s'], self.eyedata.tx / 1000.0)
        np.testing.assert_allclose(self.eyedata['time_minutes'], self.eyedata.tx / (1000.0 * 60.0))
        np.testing.assert_allclose(self.eyedata['time_hrs'], self.eyedata.tx / (1000.0 * 3600.0))
        np.testing.assert_allclose(self.eyedata['time', 'hours'], self.eyedata.tx / (1000.0 * 3600.0))

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

    def test_reset_time_defaults(self):
        """reset_time should shift time axes in-place when inplace is None."""
        data = self.eyedata.copy()
        original_tx = data.tx.copy()
        original_onsets = data.event_onsets.copy()

        result = data.reset_time()

        self.assertIs(result, data)
        expected_shift = original_tx[0]
        np.testing.assert_allclose(data.tx, original_tx - expected_shift)
        np.testing.assert_allclose(data.event_onsets, original_onsets - expected_shift)
        self.assertAlmostEqual(data.tx[0], 0.0)

    def test_reset_time_with_t0_and_copy(self):
        """reset_time should honour t0 argument and leave original untouched when inplace=False."""
        original_tx = self.eyedata.tx.copy()
        original_onsets = self.eyedata.event_onsets.copy()
        t0 = original_tx[10]

        result = self.eyedata.reset_time(t0=t0, inplace=False)

        self.assertIsNot(result, self.eyedata)
        np.testing.assert_array_equal(self.eyedata.tx, original_tx)
        np.testing.assert_array_equal(self.eyedata.event_onsets, original_onsets)
        np.testing.assert_allclose(result.tx, original_tx - t0)
        np.testing.assert_allclose(result.event_onsets, original_onsets - t0)
        self.assertAlmostEqual(result.tx[0], original_tx[0] - t0)

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
        # Use masked array mean to match what merge_eyes does
        # Note: merge_eyes uses np.ma.mean which averages non-masked values
        left_ma = self.eyedata['left', 'pupil']
        right_ma = self.eyedata['right', 'pupil']
        stacked = np.ma.stack([left_ma, right_ma], axis=0)
        expected_mean = np.ma.mean(stacked, axis=0)
        np.testing.assert_array_equal(
            merged.data['mean_pupil'],
            expected_mean.data
        )
        
        # Test merging without keeping original eyes
        merged_no_keep = self.eyedata.merge_eyes(eyes=['left', 'right'], 
                                                variables=['pupil'], keep_eyes=False)
        self.assertNotIn('left_pupil', merged_no_keep.data)
        self.assertNotIn('right_pupil', merged_no_keep.data)
        self.assertIn('mean_pupil', merged_no_keep.data)
    
    def test_merge_eyes_preserves_masks(self):
        """Test that merge_eyes handles masks correctly when averaging eyes
        
        When averaging binocular data:
        - If one eye is masked, use the other eye's value (not masked in result)
        - If both eyes are masked, result is masked
        - This allows using good data from one eye when the other has a blink
        """
        # Make a copy to avoid modifying the shared test data
        data = self.eyedata.copy()
        
        # Manually mask some points in both eyes
        data.data.mask['left_pupil'][10] = 1   # Masked in left only
        data.data.mask['left_pupil'][20] = 1   # Masked in both
        data.data.mask['right_pupil'][15] = 1  # Masked in right only
        data.data.mask['right_pupil'][20] = 1  # Masked in both
        
        # Merge eyes
        merged = data.merge_eyes(eyes=['left', 'right'], variables=['pupil'], inplace=False)
        
        # Points masked in ONE eye should NOT be masked (uses the other eye's data)
        self.assertEqual(merged.data.mask['mean_pupil'][10], 0)  # Masked in left only -> use right
        self.assertEqual(merged.data.mask['mean_pupil'][15], 0)  # Masked in right only -> use left
        
        # Points masked in BOTH eyes should be masked
        self.assertEqual(merged.data.mask['mean_pupil'][20], 1)  # Masked in both
        
        # Non-masked points should remain unmasked
        self.assertEqual(merged.data.mask['mean_pupil'][0], 0)
        self.assertEqual(merged.data.mask['mean_pupil'][5], 0)
        
        # Verify the actual values make sense (using non-masked eye data)
        # At index 10: left is masked, so mean should equal right value
        np.testing.assert_almost_equal(
            merged.data['mean_pupil'][10],
            data.data['right_pupil'][10]
        )
        # At index 15: right is masked, so mean should equal left value  
        np.testing.assert_almost_equal(
            merged.data['mean_pupil'][15],
            data.data['left_pupil'][15]
        )
    
    def test_correct_pupil_foreshortening_preserves_masks(self):
        """Test that correct_pupil_foreshortening preserves masks from blinks/artifacts"""
        # Make a copy to avoid modifying the shared test data
        data = self.eyedata.copy()
        
        # Manually mask some points (simulating blinks)
        data.data.mask['left_pupil'][10] = 1
        data.data.mask['left_pupil'][20] = 1
        data.data.mask['right_pupil'][15] = 1
        
        # Apply foreshortening correction
        corrected = data.correct_pupil_foreshortening(
            eyes=['left', 'right'], 
            inplace=False
        )
        
        # Masks should be preserved
        self.assertEqual(corrected.data.mask['left_pupil'][10], 1)
        self.assertEqual(corrected.data.mask['left_pupil'][20], 1)
        self.assertEqual(corrected.data.mask['right_pupil'][15], 1)
        
        # Non-masked points should remain unmasked
        self.assertEqual(corrected.data.mask['left_pupil'][0], 0)
        self.assertEqual(corrected.data.mask['right_pupil'][0], 0)
    
    def test_correct_pupil_foreshortening_with_store_as_preserves_masks(self):
        """Test mask preservation when storing to a different variable"""
        # Make a copy to avoid modifying the shared test data
        data = self.eyedata.copy()
        
        # Manually mask some points
        data.data.mask['left_pupil'][10] = 1
        
        # Apply correction and store as new variable
        corrected = data.correct_pupil_foreshortening(
            eyes=['left'], 
            store_as='pupil_corrected',
            inplace=False
        )
        
        # New variable should have the same mask as original
        self.assertEqual(corrected.data.mask['left_pupil_corrected'][10], 1)
        self.assertEqual(corrected.data.mask['left_pupil_corrected'][0], 0)

    def test_blinks_merge_close(self):
        """Test merging of close blinks"""
        from pypillometry.intervals import Intervals
        
        # First create some artificial blinks
        intervals = Intervals([(100, 200), (250, 350)], units=None, 
                             data_time_range=(0, len(self.eyedata.tx)),
                             sampling_rate=self.eyedata.fs)
        self.eyedata.set_blinks(intervals, eyes=['left'], variables=['pupil'])
        
        # Merge blinks that are within 100ms of each other
        merged = self.eyedata.blinks_merge_close(eyes=['left'], variables=['pupil'], distance=100, units="ms")
        blinks = merged.get_blinks('left', 'pupil')
        
        # blinks is now an Intervals object
        self.assertIsInstance(blinks, Intervals)
        self.assertEqual(len(blinks), 1)  # Should merge into one blink
        
        # Convert to integer indices for indexing
        blinks_ix = np.array(blinks.to_units("indices")).astype(int)
        self.assertEqual(blinks_ix[0, 0], 100)  # Start of first blink
        self.assertEqual(blinks_ix[0, 1], 350)  # End of last blink

        # Test merging with distance specified in seconds
        intervals2 = Intervals([(100, 200), (250, 350)], units=None,
                              data_time_range=(0, len(self.eyedata.tx)),
                              sampling_rate=self.eyedata.fs)
        self.eyedata.set_blinks(intervals2, eyes=['left'], variables=['pupil'])
        merged_sec = self.eyedata.blinks_merge_close(eyes=['left'], variables=['pupil'], distance=0.1, units="sec")
        blinks_sec = merged_sec.get_blinks('left', 'pupil')
        blinks_ix_sec = np.array(blinks_sec.to_units("indices")).astype(int)
        self.assertEqual(blinks_ix_sec[0, 0], 100)
        self.assertEqual(blinks_ix_sec[0, 1], 350)

        # Ensure blinks beyond distance remain separate
        intervals3 = Intervals([(100, 200), (250, 350)], units=None,
                              data_time_range=(0, len(self.eyedata.tx)),
                              sampling_rate=self.eyedata.fs)
        self.eyedata.set_blinks(intervals3, eyes=['left'], variables=['pupil'])
        separated = self.eyedata.blinks_merge_close(eyes=['left'], variables=['pupil'], distance=10, units="ms")
        blinks_sep = separated.get_blinks('left', 'pupil')
        blinks_ix_sep = np.array(blinks_sep.to_units("indices")).astype(int)
        self.assertEqual(len(blinks_ix_sep), 2)
        np.testing.assert_array_equal(blinks_ix_sep[0], [100, 200])
        np.testing.assert_array_equal(blinks_ix_sep[1], [250, 350])

        # Blink extending to end of signal should convert without IndexError
        end_idx = len(self.eyedata.tx) - 1
        intervals_end = Intervals([(end_idx - 10, end_idx)], units=None,
                                 data_time_range=(0, len(self.eyedata.tx)),
                                 sampling_rate=self.eyedata.fs)
        self.eyedata.set_blinks(intervals_end, eyes=['left'], variables=['pupil'])
        blinks_end = self.eyedata.get_blinks('left', 'pupil', units='ms')
        self.assertEqual(len(blinks_end), 1)
        self.assertLessEqual(blinks_end.intervals[0][1], self.eyedata.tx[-1])

    def test_blinks_merge_close_apply_mask_true(self):
        """Test that blinks_merge_close with apply_mask=True applies masks to data"""
        from pypillometry.intervals import Intervals
        
        data = self.eyedata.copy()
        
        # Create artificial blinks that are close together
        intervals = Intervals([(100, 200), (250, 350)], units=None, 
                             data_time_range=(0, len(data.tx)),
                             sampling_rate=data.fs)
        data.set_blinks(intervals, eyes=['left'], variables=['pupil'], apply_mask=False)
        
        # Verify no masks are applied initially (except any pre-existing ones)
        initial_mask_count = np.sum(data.data.mask['left_pupil'])
        
        # Merge blinks with apply_mask=True (default)
        merged = data.blinks_merge_close(eyes=['left'], variables=['pupil'], 
                                         distance=100, units="ms", apply_mask=True)
        
        # Should return the modified object
        self.assertIs(merged, data)
        
        # Verify masks are applied - the entire range from 100 to 350 should be masked
        final_mask_count = np.sum(data.data.mask['left_pupil'])
        self.assertGreater(final_mask_count, initial_mask_count)
        
        # Check that the merged region is masked
        self.assertTrue(np.all(data.data.mask['left_pupil'][100:350]))
        
        # Verify the blinks are stored
        blinks = data.get_blinks('left', 'pupil')
        self.assertEqual(len(blinks), 1)
        blinks_ix = np.array(blinks.to_units("indices")).astype(int)
        self.assertEqual(blinks_ix[0, 0], 100)
        self.assertEqual(blinks_ix[0, 1], 350)

    def test_blinks_merge_close_apply_mask_false(self):
        """Test that blinks_merge_close with apply_mask=False returns Intervals without applying masks"""
        from pypillometry.intervals import Intervals
        
        data = self.eyedata.copy()
        
        # Create artificial blinks that are close together
        intervals = Intervals([(100, 200), (250, 350)], units=None, 
                             data_time_range=(0, len(data.tx)),
                             sampling_rate=data.fs)
        data.set_blinks(intervals, eyes=['left'], variables=['pupil'], apply_mask=False)
        
        # Get initial mask state
        initial_mask = data.data.mask['left_pupil'].copy()
        
        # Merge blinks with apply_mask=False
        result = data.blinks_merge_close(eyes=['left'], variables=['pupil'], 
                                         distance=100, units="ms", apply_mask=False)
        
        # Should return a dictionary of Intervals
        self.assertIsInstance(result, dict)
        self.assertIn('left_pupil', result)
        self.assertIsInstance(result['left_pupil'], Intervals)
        
        # Verify the merged intervals are correct
        merged_intervals = result['left_pupil']
        self.assertEqual(len(merged_intervals), 1)
        merged_ix = np.array(merged_intervals.to_units("indices")).astype(int)
        self.assertEqual(merged_ix[0, 0], 100)
        self.assertEqual(merged_ix[0, 1], 350)
        
        # Verify masks were NOT applied to the data
        np.testing.assert_array_equal(data.data.mask['left_pupil'], initial_mask)
        
        # Verify stored blinks were not updated
        stored_blinks = data.get_blinks('left', 'pupil')
        self.assertEqual(len(stored_blinks), 2)  # Original two blinks still there

    def test_blinks_merge_close_multiple_eyes_apply_mask_false(self):
        """Test that apply_mask=False returns correct dictionary for multiple eyes"""
        from pypillometry.intervals import Intervals
        
        data = self.eyedata.copy()
        
        # Create blinks for both eyes
        intervals_left = Intervals([(100, 200), (250, 350)], units=None, 
                                   data_time_range=(0, len(data.tx)),
                                   sampling_rate=data.fs)
        intervals_right = Intervals([(150, 250), (400, 500)], units=None, 
                                    data_time_range=(0, len(data.tx)),
                                    sampling_rate=data.fs)
        
        data.set_blinks(intervals_left, eyes=['left'], variables=['pupil'], apply_mask=False)
        data.set_blinks(intervals_right, eyes=['right'], variables=['pupil'], apply_mask=False)
        
        # Merge blinks for both eyes with apply_mask=False
        result = data.blinks_merge_close(eyes=['left', 'right'], variables=['pupil'], 
                                         distance=100, units="ms", apply_mask=False)
        
        # Should have both eyes in the result
        self.assertIn('left_pupil', result)
        self.assertIn('right_pupil', result)
        
        # Check left eye merging (100-200 and 250-350 should merge)
        left_merged = result['left_pupil']
        self.assertEqual(len(left_merged), 1)
        left_ix = np.array(left_merged.to_units("indices")).astype(int)
        self.assertEqual(left_ix[0, 0], 100)
        self.assertEqual(left_ix[0, 1], 350)
        
        # Check right eye (150-250 and 400-500 should NOT merge as they're 150ms apart)
        right_merged = result['right_pupil']
        self.assertEqual(len(right_merged), 2)
        right_ix = np.array(right_merged.to_units("indices")).astype(int)
        np.testing.assert_array_equal(right_ix[0], [150, 250])
        np.testing.assert_array_equal(right_ix[1], [400, 500])

    def test_pupil_blinks_detect_updates_mask(self):
        """Test that blink detection updates data mask correctly."""
        # Create artificial signal with values set to 0 indicating blinks
        data_copy = self.eyedata.copy()
        data_copy.data['left_pupil'][:] = 1.0
        data_copy.data['left_pupil'][100:150] = 0.0
        data_copy.data['left_pupil'][300:360] = 0.0

        result = data_copy.pupil_blinks_detect(eyes=['left'], blink_val=0.0, units="ms")

        blinks = result.get_blinks('left', 'pupil')
        self.assertGreater(len(blinks), 0)

        mask = result.data.mask['left_pupil']
        blink_indices = np.array(blinks.to_units("indices")).astype(int)
        for start, end in blink_indices:
            self.assertTrue(np.all(mask[start:end] == 1))
    
    def test_get_blinks_returns_empty_intervals(self):
        """Test that get_blinks returns empty Intervals when no blinks detected"""
        from pypillometry.intervals import Intervals
        
        # Get blinks for an eye that has no blinks
        blinks = self.eyedata.get_blinks('left', 'pupil')
        
        self.assertIsInstance(blinks, Intervals)
        self.assertEqual(len(blinks), 0)
    
    def test_get_blinks_with_units(self):
        """Test get_blinks with units parameter"""
        from pypillometry.intervals import Intervals
        
        # Create some artificial blinks
        intervals = Intervals([(100, 200), (250, 350)], units=None,
                             data_time_range=(0, len(self.eyedata.tx)),
                             sampling_rate=self.eyedata.fs)
        self.eyedata.set_blinks(intervals, eyes=['left'], variables=['pupil'])
        
        # Get blinks in different units
        blinks_idx = self.eyedata.get_blinks('left', 'pupil', units=None)
        blinks_ms = self.eyedata.get_blinks('left', 'pupil', units='ms')
        blinks_sec = self.eyedata.get_blinks('left', 'pupil', units='sec')
        
        self.assertIsInstance(blinks_idx, Intervals)
        self.assertIsInstance(blinks_ms, Intervals)
        self.assertIsInstance(blinks_sec, Intervals)
        
        self.assertIsNone(blinks_idx.units)
        self.assertEqual(blinks_ms.units, 'ms')
        self.assertEqual(blinks_sec.units, 'sec')
        
        self.assertEqual(len(blinks_idx), 2)
        self.assertEqual(len(blinks_ms), 2)
        
        # Test with aliases
        blinks_seconds = self.eyedata.get_blinks('left', 'pupil', units='seconds')
        blinks_s = self.eyedata.get_blinks('left', 'pupil', units='s')
        blinks_minutes = self.eyedata.get_blinks('left', 'pupil', units='minutes')
        
        self.assertEqual(blinks_seconds.units, 'sec')
        self.assertEqual(blinks_s.units, 'sec')
        self.assertEqual(blinks_minutes.units, 'min')
        self.assertEqual(len(blinks_seconds), 2)
        self.assertEqual(len(blinks_s), 2)
        self.assertEqual(len(blinks_sec), 2)
    
    def test_get_blinks_merged_multiple_eyes(self):
        """Test getting blinks for multiple eyes returns a dict"""
        from pypillometry.intervals import Intervals
        
        # Create blinks for both eyes
        intervals_left = Intervals([(100, 200)], units=None,
                                  data_time_range=(0, len(self.eyedata.tx)),
                                  sampling_rate=self.eyedata.fs)
        intervals_right = Intervals([(150, 250)], units=None,
                                   data_time_range=(0, len(self.eyedata.tx)),
                                   sampling_rate=self.eyedata.fs)
        self.eyedata.set_blinks(intervals_left, eyes=['left'], variables=['pupil'])
        self.eyedata.set_blinks(intervals_right, eyes=['right'], variables=['pupil'])
        
        # Get blinks for multiple eyes - should return a dict
        blinks = self.eyedata.get_blinks(['left', 'right'], 'pupil')
        
        self.assertIsInstance(blinks, dict)
        self.assertIn('left_pupil', blinks)
        self.assertIn('right_pupil', blinks)
        self.assertIsInstance(blinks['left_pupil'], Intervals)
        self.assertIsInstance(blinks['right_pupil'], Intervals)

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
        
        # Test with aliases
        duration_seconds = self.eyedata.get_duration(units='seconds')
        duration_s = self.eyedata.get_duration(units='s')
        duration_minutes = self.eyedata.get_duration(units='minutes')
        duration_hrs = self.eyedata.get_duration(units='hrs')
        
        self.assertAlmostEqual(duration_sec, duration_seconds)
        self.assertAlmostEqual(duration_sec, duration_s)
        self.assertAlmostEqual(duration_min, duration_minutes)
        self.assertAlmostEqual(duration_h, duration_hrs)

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
    
    def test_mask_intervals_single(self):
        """Test mask_intervals with a single Intervals object"""
        from pypillometry.intervals import Intervals
        
        # Create test intervals
        intervals = Intervals([(100, 200), (300, 400)], units=None,
                             data_time_range=(0, len(self.eyedata.tx)),
                             sampling_rate=self.eyedata.fs)
        
        # Apply intervals as mask to left eye pupil
        result = self.eyedata.mask_intervals(intervals, eyes=['left'], variables=['pupil'])
        
        # Check that mask was applied
        mask = result.data.mask['left_pupil']
        self.assertTrue(np.all(mask[100:200] == 1))
        self.assertTrue(np.all(mask[300:400] == 1))
        self.assertTrue(np.all(mask[0:100] == 0))
        
    def test_mask_intervals_dict(self):
        """Test mask_intervals with a dict of Intervals"""
        from pypillometry.intervals import Intervals
        
        # Create intervals for different variables
        intervals_dict = {
            'left_pupil': Intervals([(100, 200)], units=None,
                                   data_time_range=(0, len(self.eyedata.tx)),
                                   sampling_rate=self.eyedata.fs),
            'right_pupil': Intervals([(150, 250)], units=None,
                                    data_time_range=(0, len(self.eyedata.tx)),
                                    sampling_rate=self.eyedata.fs)
        }
        
        # Apply intervals
        result = self.eyedata.mask_intervals(intervals_dict)
        
        # Check that masks were applied correctly
        self.assertTrue(np.all(result.data.mask['left_pupil'][100:200] == 1))
        self.assertTrue(np.all(result.data.mask['right_pupil'][150:250] == 1))
        
    def test_pupil_blinks_detect_apply_mask_false(self):
        """Test pupil_blinks_detect with apply_mask=False returns Intervals"""
        from pypillometry.intervals import Intervals
        
        # Create artificial signal with blinks
        data_copy = self.eyedata.copy()
        data_copy.data['left_pupil'][:] = 1.0
        data_copy.data['left_pupil'][100:150] = 0.0
        
        # Detect without applying mask
        result = data_copy.pupil_blinks_detect(eyes=['left'], blink_val=0.0, 
                                              apply_mask=False, units="ms")
        
        # Should return Intervals object (single eye)
        self.assertIsInstance(result, Intervals)
        self.assertGreater(len(result), 0)
        
        # Mask should NOT be applied yet
        self.assertFalse(np.all(data_copy.data.mask['left_pupil'][100:150] == 1))
        
    def test_pupil_blinks_detect_apply_mask_false_multiple_eyes(self):
        """Test pupil_blinks_detect with apply_mask=False and multiple eyes returns dict"""
        from pypillometry.intervals import Intervals
        
        # Create artificial signal with blinks
        data_copy = self.eyedata.copy()
        data_copy.data['left_pupil'][:] = 1.0
        data_copy.data['left_pupil'][100:150] = 0.0
        data_copy.data['right_pupil'][:] = 1.0
        data_copy.data['right_pupil'][200:250] = 0.0
        
        # Detect without applying mask
        result = data_copy.pupil_blinks_detect(eyes=['left', 'right'], blink_val=0.0,
                                              apply_mask=False, units="ms")
        
        # Should return dict of Intervals
        self.assertIsInstance(result, dict)
        self.assertIn('left_pupil', result)
        self.assertIn('right_pupil', result)
        self.assertIsInstance(result['left_pupil'], Intervals)
        self.assertIsInstance(result['right_pupil'], Intervals)

    def test_setitem_masked_array(self):
        """Test __setitem__ with masked array"""
        data = {
            'left_pupil': np.array([1, 2, 3, 4, 5]),
            'right_pupil': np.array([1, 2, 3, 4, 5]),
            'left_x': np.array([1, 2, 3, 4, 5]),
            'left_y': np.array([1, 2, 3, 4, 5]),
            'right_x': np.array([1, 2, 3, 4, 5]),
            'right_y': np.array([1, 2, 3, 4, 5])
        }
        
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


class TestEyeDataSetitem(unittest.TestCase):
    """Test __setitem__ accessor for setting data arrays"""
    
    def setUp(self):
        """Set up test data"""
        self.eyedata = get_rlmw_002_short()
        self.original_tx = self.eyedata.tx.copy()
        self.original_left_pupil = self.eyedata.data['left_pupil'].copy()
    
    def test_setitem_regular_array(self):
        """Setting a regular numpy array should work and create zero mask"""
        new_data = np.array([100.0, 200.0, 300.0] + [400.0] * (len(self.eyedata) - 3))
        self.eyedata['left_pupil'] = new_data
        
        np.testing.assert_array_equal(self.eyedata.data['left_pupil'], new_data)
        # Mask should be all zeros (no masked values)
        np.testing.assert_array_equal(
            self.eyedata.data.mask['left_pupil'], 
            np.zeros(len(new_data), dtype=int)
        )
    
    def test_setitem_masked_array(self):
        """Setting a masked array should preserve both data and mask"""
        data_part = np.array([100.0, 200.0, 300.0] + [400.0] * (len(self.eyedata) - 3))
        mask_part = np.array([0, 1, 0] + [0] * (len(self.eyedata) - 3), dtype=bool)
        masked_arr = np.ma.masked_array(data_part, mask=mask_part)
        
        self.eyedata['left_pupil'] = masked_arr
        
        np.testing.assert_array_equal(self.eyedata.data['left_pupil'], data_part)
        np.testing.assert_array_equal(
            self.eyedata.data.mask['left_pupil'], 
            mask_part.astype(int)
        )
    
    def test_setitem_with_tuple_key(self):
        """Setting with tuple key should work like string key"""
        new_data = np.array([500.0, 600.0] + [700.0] * (len(self.eyedata) - 2))
        self.eyedata['left', 'pupil'] = new_data
        
        np.testing.assert_array_equal(self.eyedata.data['left_pupil'], new_data)
    
    def test_setitem_time_default_ms(self):
        """Setting 'time' should update tx (assumes milliseconds)"""
        new_time = np.arange(len(self.eyedata)) * 2.0  # 0, 2, 4, 6, ...
        self.eyedata['time'] = new_time
        
        np.testing.assert_array_equal(self.eyedata.tx, new_time)
    
    def test_setitem_time_with_sec_conversion(self):
        """Setting 'time_sec' should convert from seconds to milliseconds"""
        new_time_sec = np.arange(len(self.eyedata)) * 0.002  # 0, 0.002, 0.004, ...
        self.eyedata['time_sec'] = new_time_sec
        
        expected_ms = new_time_sec * 1000.0
        np.testing.assert_allclose(self.eyedata.tx, expected_ms)
    
    def test_setitem_time_with_min_conversion(self):
        """Setting 'time_min' should convert from minutes to milliseconds"""
        new_time_min = np.arange(len(self.eyedata)) / 30000.0  # Small fractions of minutes
        self.eyedata['time_min'] = new_time_min
        
        expected_ms = new_time_min * 60.0 * 1000.0
        np.testing.assert_allclose(self.eyedata.tx, expected_ms)
    
    def test_setitem_time_with_h_conversion(self):
        """Setting 'time_h' should convert from hours to milliseconds"""
        new_time_h = np.arange(len(self.eyedata)) / 1800000.0  # Tiny fractions of hours
        self.eyedata['time_h'] = new_time_h
        
        expected_ms = new_time_h * 3600.0 * 1000.0
        np.testing.assert_allclose(self.eyedata.tx, expected_ms)
    
    def test_setitem_time_with_tuple_key(self):
        """Setting ('time', 'sec') should work like 'time_sec'"""
        new_time_sec = np.arange(len(self.eyedata)) * 0.002
        self.eyedata['time', 'sec'] = new_time_sec
        
        expected_ms = new_time_sec * 1000.0
        np.testing.assert_allclose(self.eyedata.tx, expected_ms)
    
    def test_setitem_time_with_aliases(self):
        """Setting time with unit aliases should work"""
        # Test "seconds" alias
        new_time = np.arange(len(self.eyedata)) * 0.002
        self.eyedata['time_seconds'] = new_time
        np.testing.assert_allclose(self.eyedata.tx, new_time * 1000.0)
        
        # Test "s" alias
        self.eyedata['time_s'] = new_time
        np.testing.assert_allclose(self.eyedata.tx, new_time * 1000.0)
        
        # Test "minutes" alias
        new_time_min = np.arange(len(self.eyedata)) / 30000.0
        self.eyedata['time_minutes'] = new_time_min
        np.testing.assert_allclose(self.eyedata.tx, new_time_min * 60000.0)
        
        # Test "hrs" alias
        new_time_h = np.arange(len(self.eyedata)) / 1800000.0
        self.eyedata['time_hrs'] = new_time_h
        np.testing.assert_allclose(self.eyedata.tx, new_time_h * 3600000.0)
    
    def test_setitem_new_variable(self):
        """Setting a new variable should add it to the data dict"""
        new_var = np.ones(len(self.eyedata)) * 999.0
        self.eyedata['left_new_variable'] = new_var
        
        self.assertIn('left_new_variable', self.eyedata.data)
        np.testing.assert_array_equal(self.eyedata.data['left_new_variable'], new_var)
    
    def test_setitem_retrieval_roundtrip(self):
        """Setting then getting should return the same masked array"""
        data_part = np.array([111.0, 222.0] + [333.0] * (len(self.eyedata) - 2))
        mask_part = np.array([1, 0] + [0] * (len(self.eyedata) - 2), dtype=bool)
        masked_arr = np.ma.masked_array(data_part, mask=mask_part)
        
        self.eyedata['right_pupil'] = masked_arr
        retrieved = self.eyedata['right_pupil']
        
        self.assertIsInstance(retrieved, np.ma.MaskedArray)
        np.testing.assert_array_equal(retrieved.data, data_part)
        np.testing.assert_array_equal(retrieved.mask, mask_part)


class TestEyeDataDelitem(unittest.TestCase):
    """Test __delitem__ accessor for deleting data arrays"""
    
    def setUp(self):
        """Set up test data"""
        self.eyedata = get_rlmw_002_short()
    
    def test_delitem_string_key(self):
        """Deleting with string key should remove data and mask"""
        # Verify it exists first
        self.assertIn('left_pupil', self.eyedata.data)
        
        # Delete it
        del self.eyedata['left_pupil']
        
        # Verify it's gone
        self.assertNotIn('left_pupil', self.eyedata.data)
        self.assertNotIn('left_pupil', self.eyedata.data.mask)
    
    def test_delitem_tuple_key(self):
        """Deleting with tuple key should work like string key"""
        # Verify it exists first
        self.assertIn('right_x', self.eyedata.data)
        
        # Delete with tuple key
        del self.eyedata['right', 'x']
        
        # Verify it's gone
        self.assertNotIn('right_x', self.eyedata.data)
    
    def test_delitem_nonexistent_key_raises(self):
        """Deleting a nonexistent key should raise KeyError"""
        with self.assertRaises(KeyError):
            del self.eyedata['nonexistent_key']
    
    def test_delitem_time_raises_error(self):
        """Deleting time array should raise KeyError (time not in data dict)"""
        with self.assertRaises(KeyError):
            del self.eyedata['time']
    
    def test_delitem_time_with_unit_raises_error(self):
        """Deleting time_* keys should raise KeyError (not in data dict)"""
        with self.assertRaises(KeyError):
            del self.eyedata['time_sec']
        
        with self.assertRaises(KeyError):
            del self.eyedata['time_ms']
    
    def test_delitem_time_tuple_raises_error(self):
        """Deleting ('time', 'sec') should raise ValueError"""
        with self.assertRaises(ValueError) as cm:
            del self.eyedata['time', 'sec']
        self.assertIn("Cannot delete time array", str(cm.exception))
    
    def test_delitem_empty_tuple_raises(self):
        """Deleting with empty tuple should raise KeyError"""
        with self.assertRaises(KeyError) as cm:
            del self.eyedata[()]
        self.assertIn("Empty key tuple", str(cm.exception))
    
    def test_delitem_then_setitem(self):
        """After deleting, should be able to set a new value"""
        # Delete
        del self.eyedata['left_x']
        self.assertNotIn('left_x', self.eyedata.data)
        
        # Set new value
        new_data = np.ones(len(self.eyedata)) * 555.0
        self.eyedata['left_x'] = new_data
        
        # Verify it's back
        self.assertIn('left_x', self.eyedata.data)
        np.testing.assert_array_equal(self.eyedata.data['left_x'], new_data)
    
    def test_pupil_blinks_interpolate_preserves_mask(self):
        """Test that blink interpolation preserves original mask and marks interpolated regions"""
        # Create pupil data with some manually masked regions
        pd = pp.PupilData(
            left_pupil=np.random.rand(1000) * 100 + 3000,
            sampling_rate=100
        )
        
        # Manually mask some regions (not blinks)
        manual_mask_indices = slice(100, 120)
        pd.data.mask['left_pupil'][manual_mask_indices] = 1
        
        # Add some blinks (zeros) - need enough zeros to be detected as blinks
        blink_indices = slice(500, 520)  # 20 samples at 100Hz = 200ms
        pd.data['left_pupil'][blink_indices] = 0
        
        # Store original mask
        original_mask = pd.data.mask['left_pupil'].copy()
        
        # First detect blinks, then interpolate
        pd = pd.pupil_blinks_detect(eyes=['left'], min_duration=50, inplace=True)
        pd_interp = pd.pupil_blinks_interpolate(eyes=['left'], store_as='pupil', inplace=False)
        
        # Verify manual mask is preserved
        np.testing.assert_array_equal(
            pd_interp.data.mask['left_pupil'][manual_mask_indices],
            original_mask[manual_mask_indices],
            err_msg="Manually masked regions should be preserved"
        )
        
        # Verify blink regions are now masked (interpolated regions are marked)
        # Note: not all indices might be interpolated depending on the algorithm
        self.assertTrue(
            np.any(pd_interp.data.mask['left_pupil'][blink_indices] == 1),
            "At least some interpolated blink regions should be masked"
        )
        
        # Verify data was interpolated (not all zeros anymore)
        self.assertTrue(
            np.any(pd_interp.data['left_pupil'][blink_indices] > 0),
            "Blink regions should be interpolated (not all zero)"
        )
    
    def test_pupil_lowpass_preserves_mask(self):
        """Test that lowpass filtering preserves existing mask"""
        # Create pupil data
        pd = pp.PupilData(
            left_pupil=np.random.rand(1000) * 100 + 3000,
            sampling_rate=100
        )
        
        # Manually mask some regions
        mask_indices = slice(200, 250)
        pd.data.mask['left_pupil'][mask_indices] = 1
        original_mask = pd.data.mask['left_pupil'].copy()
        
        # Apply lowpass filter
        pd_filtered = pd.pupil_lowpass_filter(cutoff=10, order=4, eyes=['left'], inplace=False)
        
        # Verify mask is preserved
        np.testing.assert_array_equal(
            pd_filtered.data.mask['left_pupil'],
            original_mask,
            err_msg="Lowpass filtering should preserve the mask"
        )
        
        # Verify data was actually filtered (changed)
        self.assertFalse(
            np.array_equal(pd.data['left_pupil'], pd_filtered.data['left_pupil']),
            "Data should be filtered (changed)"
        )
    
    def test_pupil_smooth_preserves_mask(self):
        """Test that smoothing preserves existing mask"""
        # Create pupil data
        pd = pp.PupilData(
            left_pupil=np.random.rand(1000) * 100 + 3000,
            sampling_rate=100
        )
        
        # Manually mask some regions
        mask_indices = slice(300, 350)
        pd.data.mask['left_pupil'][mask_indices] = 1
        original_mask = pd.data.mask['left_pupil'].copy()
        
        # Apply smoothing
        pd_smoothed = pd.pupil_smooth_window(eyes=['left'], window='hanning', winsize=50, inplace=False)
        
        # Verify mask is preserved
        np.testing.assert_array_equal(
            pd_smoothed.data.mask['left_pupil'],
            original_mask,
            err_msg="Smoothing should preserve the mask"
        )
        
        # Verify data was actually smoothed (changed)
        self.assertFalse(
            np.array_equal(pd.data['left_pupil'], pd_smoothed.data['left_pupil']),
            "Data should be smoothed (changed)"
        )


class TestEventsIntegration(unittest.TestCase):
    """Test get_events() and set_events() methods"""
    
    def setUp(self):
        """Set up test data"""
        self.data = get_rlmw_002_short()
        self.original_event_count = self.data.nevents()
    
    def test_get_events_returns_events_object(self):
        """Test that get_events returns an Events object"""
        from pypillometry.events import Events
        
        events = self.data.get_events()
        
        self.assertIsInstance(events, Events)
        self.assertEqual(len(events), self.original_event_count)
    
    def test_get_events_default_units(self):
        """Test that get_events defaults to ms units"""
        events = self.data.get_events()
        
        self.assertEqual(events.units, "ms")
        np.testing.assert_array_equal(events.onsets, self.data.event_onsets)
        np.testing.assert_array_equal(events.labels, self.data.event_labels)
    
    def test_get_events_with_different_units(self):
        """Test get_events with different units"""
        events_ms = self.data.get_events(units="ms")
        events_sec = self.data.get_events(units="sec")
        events_min = self.data.get_events(units="min")
        
        self.assertEqual(events_sec.units, "sec")
        self.assertEqual(events_min.units, "min")
        
        # Check conversion accuracy
        np.testing.assert_array_almost_equal(
            events_sec.onsets, events_ms.onsets / 1000.0, decimal=6
        )
        np.testing.assert_array_almost_equal(
            events_min.onsets, events_ms.onsets / 60000.0, decimal=9
        )
        
        # Test with aliases
        events_seconds = self.data.get_events(units="seconds")
        events_s = self.data.get_events(units="s")
        events_minutes = self.data.get_events(units="minutes")
        events_hrs = self.data.get_events(units="hrs")
        
        self.assertEqual(events_seconds.units, "sec")
        self.assertEqual(events_s.units, "sec")
        self.assertEqual(events_minutes.units, "min")
        self.assertEqual(events_hrs.units, "h")
        
        np.testing.assert_array_almost_equal(
            events_seconds.onsets, events_sec.onsets, decimal=9
        )
        np.testing.assert_array_almost_equal(
            events_s.onsets, events_sec.onsets, decimal=9
        )
    
    def test_get_events_has_time_range(self):
        """Test that get_events includes data time range"""
        events = self.data.get_events()
        
        self.assertIsNotNone(events.data_time_range)
        self.assertAlmostEqual(events.data_time_range[0], self.data.tx[0])
        self.assertAlmostEqual(events.data_time_range[1], self.data.tx[-1])
    
    def test_set_events_round_trip(self):
        """Test that get_events -> set_events preserves data"""
        events = self.data.get_events()
        original_onsets = self.data.event_onsets.copy()
        original_labels = self.data.event_labels.copy()
        
        # Set the same events back
        result = self.data.set_events(events)
        
        np.testing.assert_array_equal(result.event_onsets, original_onsets)
        np.testing.assert_array_equal(result.event_labels, original_labels)
        self.assertIs(result, self.data)
    
    def test_set_events_with_filtered_events(self):
        """Test setting filtered events"""
        events = self.data.get_events()
        filtered = events.filter("F")
        
        result = self.data.set_events(filtered)
        
        self.assertEqual(result.nevents(), len(filtered))
        np.testing.assert_array_equal(result.event_onsets, filtered.onsets)
        np.testing.assert_array_equal(result.event_labels, filtered.labels)
        # Should return self (inplace by default)
        self.assertIs(result, self.data)
    
    def test_set_events_with_different_units(self):
        """Test that set_events handles unit conversion"""
        # Get events in seconds
        events_sec = self.data.get_events(units="sec")
        original_onsets_ms = self.data.event_onsets.copy()
        
        # Set them back (should convert to ms internally)
        result = self.data.set_events(events_sec)
        
        # Should match original (within floating point precision)
        np.testing.assert_array_almost_equal(
            result.event_onsets, original_onsets_ms, decimal=3
        )
        self.assertIs(result, self.data)
    
    def test_set_events_invalid_type(self):
        """Test that set_events rejects non-Events objects"""
        with self.assertRaises(TypeError):
            self.data.set_events([100, 200, 300])
        
        with self.assertRaises(TypeError):
            self.data.set_events({"onsets": [100, 200], "labels": ["A", "B"]})
    
    def test_get_events_empty(self):
        """Test get_events with no events"""
        # Create data with no events
        data = pp.EyeData(
            left_x=[1, 2, 3],
            left_y=[4, 5, 6],
            left_pupil=[7, 8, 9],
            sampling_rate=1000,
            event_onsets=None,
            event_labels=None
        )
        
        events = data.get_events()
        
        self.assertEqual(len(events), 0)
        self.assertEqual(events.units, "ms")
    
    def test_set_events_empty(self):
        """Test setting empty events"""
        from pypillometry.events import Events
        
        empty_events = Events([], [], units="ms")
        result = self.data.set_events(empty_events)
        
        self.assertEqual(result.nevents(), 0)
        self.assertIs(result, self.data)
    
    def test_filter_and_set_workflow(self):
        """Test typical workflow: get -> filter -> set"""
        # Get events, filter them, and set back
        events = self.data.get_events()
        stim_events = events.filter("F")
        
        original_count = len(events)
        filtered_count = len(stim_events)
        
        self.assertLess(filtered_count, original_count)
        
        # Set filtered events
        result = self.data.set_events(stim_events)
        self.assertEqual(result.nevents(), filtered_count)
        self.assertIs(result, self.data)
        
        # Verify all events match the filter
        for label in result.event_labels:
            self.assertIn("F", label)
    
    def test_get_events_labels_preserved(self):
        """Test that event labels are properly preserved"""
        events = self.data.get_events()
        
        # All labels should be strings
        for label in events.labels:
            self.assertIsInstance(label, (str, np.str_))
        
        # Should match original labels
        for orig_label, event_label in zip(self.data.event_labels, events.labels):
            self.assertEqual(str(orig_label), str(event_label))
    
    def test_set_events_inplace_parameter(self):
        """Test set_events with inplace parameter"""
        events = self.data.get_events()
        filtered = events.filter("F")
        
        # Test inplace=True
        data_copy1 = self.data.copy()
        result1 = data_copy1.set_events(filtered, inplace=True)
        self.assertIs(result1, data_copy1)
        self.assertEqual(result1.nevents(), len(filtered))
        
        # Test inplace=False
        data_copy2 = self.data.copy()
        original_count = data_copy2.nevents()
        result2 = data_copy2.set_events(filtered, inplace=False)
        self.assertIsNot(result2, data_copy2)
        self.assertEqual(data_copy2.nevents(), original_count)  # Original unchanged
        self.assertEqual(result2.nevents(), len(filtered))  # Result has filtered events
    
    def test_set_events_keeps_history(self):
        """Test that set_events adds to history"""
        events = self.data.get_events()
        filtered = events.filter("F")
        
        original_history_len = len(self.data.history)
        self.data.set_events(filtered)
        
        # History should have increased
        self.assertGreater(len(self.data.history), original_history_len)
        # Last entry should be set_events
        self.assertEqual(self.data.history[-1]["funcname"], "set_events")


class TestEventsWithGetIntervals(unittest.TestCase):
    """Test get_intervals() with Events objects"""
    
    def setUp(self):
        """Set up test data"""
        self.data = get_rlmw_002_short()
    
    def test_get_intervals_with_events_object(self):
        """Test that get_intervals accepts Events object"""
        from pypillometry.intervals import Intervals
        
        events = self.data.get_events()
        intervals = self.data.get_intervals(events, interval=(-200, 200), units="ms")
        
        self.assertIsInstance(intervals, Intervals)
        self.assertEqual(len(intervals), len(events))
        self.assertEqual(intervals.units, "ms")
    
    def test_get_intervals_with_filtered_events(self):
        """Test get_intervals with filtered Events object"""
        events = self.data.get_events()
        filtered = events.filter("F")
        
        intervals = self.data.get_intervals(filtered, interval=(-200, 200), units="ms")
        
        # Should have same number of intervals as filtered events
        self.assertEqual(len(intervals), len(filtered))
        
        # Compare with string selector
        intervals_str = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        self.assertEqual(len(intervals), len(intervals_str))
    
    def test_get_intervals_events_vs_string(self):
        """Test that Events and string selector give same results"""
        events = self.data.get_events()
        filtered = events.filter("F")
        
        intervals_events = self.data.get_intervals(filtered, interval=(-200, 200), units="ms")
        intervals_str = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        
        # Should have same number of intervals
        self.assertEqual(len(intervals_events), len(intervals_str))
        
        # Should have same event labels
        self.assertEqual(intervals_events.event_labels, intervals_str.event_labels)
    
    def test_get_intervals_with_events_different_units(self):
        """Test get_intervals with Events in different units"""
        events = self.data.get_events()
        events_sec = events.to_units("sec")
        
        intervals = self.data.get_intervals(events_sec, interval=(-0.2, 0.2), units="sec")
        
        self.assertEqual(len(intervals), len(events))
        self.assertEqual(intervals.units, "sec")
    
    def test_get_intervals_with_events_custom_label(self):
        """Test that custom label works with Events object"""
        events = self.data.get_events()
        intervals = self.data.get_intervals(events, interval=(-200, 200), units="ms", label="custom_label")
        
        self.assertEqual(intervals.label, "custom_label")
    
    def test_get_intervals_with_events_automatic_label(self):
        """Test that automatic label is generated when not provided"""
        events = self.data.get_events()
        filtered = events.filter("F")
        
        intervals = self.data.get_intervals(filtered, interval=(-200, 200), units="ms")
        
        # Should have automatic label
        self.assertIsNotNone(intervals.label)
        self.assertIn("events", intervals.label.lower())
    
    def test_get_intervals_with_events_units_consistency(self):
        """Test that units are consistent between Events and resulting Intervals"""
        events = self.data.get_events(units="sec")
        
        intervals = self.data.get_intervals(events, interval=(-0.2, 0.2), units="sec")
        
        self.assertEqual(intervals.units, "sec")
        self.assertEqual(events.units, "sec")
    
    def test_get_intervals_with_events_indices(self):
        """Test get_intervals with Events using index units"""
        events = self.data.get_events()
        
        intervals = self.data.get_intervals(events, interval=(-10, 10), units=None)
        
        self.assertEqual(intervals.units, None)
        self.assertEqual(len(intervals), len(events))
    
    def test_get_intervals_with_empty_events(self):
        """Test get_intervals with empty Events object"""
        from pypillometry.events import Events
        
        empty_events = Events([], [], units="ms")
        
        intervals = self.data.get_intervals(empty_events, interval=(-200, 200), units="ms")
        
        self.assertEqual(len(intervals), 0)
    
    def test_get_intervals_events_preserves_labels(self):
        """Test that event labels are preserved in intervals"""
        events = self.data.get_events()
        filtered = events.filter("F")
        
        intervals = self.data.get_intervals(filtered, interval=(-200, 200), units="ms")
        
        # Should have event labels
        self.assertIsNotNone(intervals.event_labels)
        self.assertEqual(len(intervals.event_labels), len(filtered))
        
        # Labels should match
        for interval_label, event_label in zip(intervals.event_labels, filtered.labels):
            self.assertEqual(interval_label, event_label)
    
    def test_get_intervals_backward_compatibility(self):
        """Test that existing string/function selectors still work"""
        # String selector
        intervals_str = self.data.get_intervals("F", interval=(-200, 200), units="ms")
        self.assertIsNotNone(intervals_str)
        
        # Function selector
        intervals_func = self.data.get_intervals(lambda label: "F" in label, interval=(-200, 200), units="ms")
        self.assertIsNotNone(intervals_func)
        
        # Should have same number of intervals
        self.assertEqual(len(intervals_str), len(intervals_func))

    def test_mask_as_intervals_single_eye_variable(self):
        """Test mask_as_intervals with single eye/variable"""
        data = self.data.copy()
        data.pupil_blinks_detect()
        
        # Test single eye/variable returns Intervals object
        intervals = data.mask_as_intervals('left', 'pupil')
        self.assertEqual(intervals.__class__.__name__, 'Intervals')
        self.assertIsNone(intervals.units)  # Default is indices
        self.assertEqual(intervals.label, 'left_pupil_mask')
        self.assertEqual(intervals.data_time_range, (0, len(data.tx)))
        
    def test_mask_as_intervals_multiple_eyes(self):
        """Test mask_as_intervals with multiple eyes returns dict"""
        data = self.data.copy()
        data.pupil_blinks_detect()
        
        # Test multiple eyes returns dict
        intervals_dict = data.mask_as_intervals(['left', 'right'], 'pupil')
        self.assertIsInstance(intervals_dict, dict)
        self.assertIn('left_pupil', intervals_dict)
        self.assertIn('right_pupil', intervals_dict)
        
        # Each value should be an Intervals object
        for key, intervals in intervals_dict.items():
            self.assertEqual(intervals.__class__.__name__, 'Intervals')
            self.assertIsNone(intervals.units)
            
    def test_mask_as_intervals_multiple_variables(self):
        """Test mask_as_intervals with multiple variables returns dict"""
        data = self.data.copy()
        data.pupil_blinks_detect()
        
        # Test multiple variables returns dict
        intervals_dict = data.mask_as_intervals('left', ['pupil', 'x'])
        self.assertIsInstance(intervals_dict, dict)
        self.assertIn('left_pupil', intervals_dict)
        self.assertIn('left_x', intervals_dict)
        
    def test_mask_as_intervals_with_units_ms(self):
        """Test mask_as_intervals with units='ms'"""
        data = self.data.copy()
        data.pupil_blinks_detect()
        
        intervals = data.mask_as_intervals('left', 'pupil', units='ms')
        self.assertEqual(intervals.units, 'ms')
        # Use approximate comparison since sampling_rate conversion may differ slightly
        self.assertAlmostEqual(intervals.data_time_range[0], data.tx[0], delta=10)
        self.assertAlmostEqual(intervals.data_time_range[1], data.tx[-1], delta=10)
        
        # If there are intervals, check they're in milliseconds
        if len(intervals) > 0:
            # Values should be larger than indices
            first_interval = intervals.intervals[0]
            self.assertGreater(first_interval[0], 10)  # Should be in ms range
            
    def test_mask_as_intervals_with_units_sec(self):
        """Test mask_as_intervals with units='sec'"""
        data = self.data.copy()
        data.pupil_blinks_detect()
        
        intervals = data.mask_as_intervals('left', 'pupil', units='sec')
        self.assertEqual(intervals.units, 'sec')
        
    def test_mask_as_intervals_with_custom_label(self):
        """Test mask_as_intervals with custom label"""
        data = self.data.copy()
        data.pupil_blinks_detect()
        
        intervals = data.mask_as_intervals('left', 'pupil', label='my_custom_label')
        self.assertEqual(intervals.label, 'my_custom_label')
        
    def test_mask_as_intervals_empty_mask(self):
        """Test mask_as_intervals with empty mask (no masked data)"""
        # Create data without zero pupil values to test truly empty mask
        data = pp.EyeData(
            left_x=np.arange(100),
            left_y=np.arange(100),
            left_pupil=np.ones(100) * 500,  # No zeros, all valid pupil sizes
            sampling_rate=1000
        )
        # Don't detect blinks, so mask should be all zeros (no zero pupil values)
        
        intervals = data.mask_as_intervals('left', 'pupil')
        self.assertEqual(len(intervals), 0)
        self.assertEqual(intervals.intervals, [])
        
    def test_mask_as_intervals_nonexistent_key(self):
        """Test mask_as_intervals with non-existent key"""
        data = self.data.copy()
        
        # Should return empty Intervals if key doesn't exist
        intervals = data.mask_as_intervals('nonexistent', 'variable')
        self.assertEqual(len(intervals), 0)
        
    def test_mask_as_intervals_with_actual_mask(self):
        """Test mask_as_intervals correctly identifies masked intervals"""
        data = self.data.copy()
        
        # Manually create a mask with known intervals
        mask = np.zeros(len(data.tx), dtype=int)
        mask[100:150] = 1  # First interval (indices 100-149)
        mask[200:250] = 1  # Second interval (indices 200-249)
        data.data.set_mask('left_pupil', mask)
        
        intervals = data.mask_as_intervals('left', 'pupil')
        self.assertEqual(len(intervals), 2)
        
        # Check intervals match what we set
        # Note: intervals use exclusive end indices (like Python slicing)
        # mask[100:150] sets indices 100-149, returned as (100, 150)
        self.assertEqual(intervals.intervals[0], (100, 150))
        self.assertEqual(intervals.intervals[1], (200, 250))
        
    def test_mask_as_intervals_multiple_eyes_with_units(self):
        """Test mask_as_intervals with multiple eyes and units"""
        data = self.data.copy()
        data.pupil_blinks_detect()
        
        intervals_dict = data.mask_as_intervals(['left', 'right'], 'pupil', units='ms')
        
        for key, intervals in intervals_dict.items():
            self.assertEqual(intervals.units, 'ms')
            # Use approximate comparison since sampling_rate conversion may differ slightly
            self.assertAlmostEqual(intervals.data_time_range[0], data.tx[0], delta=10)
            self.assertAlmostEqual(intervals.data_time_range[1], data.tx[-1], delta=10)
            
    def test_mask_as_intervals_all_eyes_all_variables(self):
        """Test mask_as_intervals with default (all eyes and variables)"""
        data = self.data.copy()
        data.pupil_blinks_detect()
        
        # Empty lists should get all available eyes and variables
        intervals_dict = data.mask_as_intervals([], [])
        self.assertIsInstance(intervals_dict, dict)
        
        # Should have entries for multiple combinations
        self.assertGreater(len(intervals_dict), 1)
    
    def test_set_experimental_setup_with_string_units(self):
        """Test set_experimental_setup with string format units"""
        data = pp.EyeData(left_x=[1,2], left_y=[3,4], left_pupil=[5,6], sampling_rate=1000)
        
        # Set with string format - eye frame for direct r value
        data.set_experimental_setup(
            camera_spherical=("20 deg", "-90 deg", "60 cm"),
            eye_to_screen_center="70 cm",
            physical_screen_size=("52 cm", "29 cm")
        )
        
        # All should be stored in mm
        self.assertEqual(data.experimental_setup.r, 600.0)
        self.assertEqual(data.experimental_setup.d, 700.0)
        self.assertEqual(data.experimental_setup.physical_screen_width, 520.0)
        self.assertEqual(data.experimental_setup.physical_screen_height, 290.0)
    
    def test_set_experimental_setup_with_pint_quantities(self):
        """Test set_experimental_setup with Pint Quantities"""
        data = pp.EyeData(left_x=[1,2], left_y=[3,4], left_pupil=[5,6], sampling_rate=1000)
        
        # Set with Pint Quantities - eye frame for direct r value
        data.set_experimental_setup(
            camera_spherical=(20 * pp.ureg.deg, -90 * pp.ureg.deg, 60 * pp.ureg.cm),
            ipd=65 * pp.ureg.mm,
            eye_to_screen_center=70 * pp.ureg.cm
        )
        
        # All should be stored in mm
        self.assertEqual(data.experimental_setup.r, 600.0)
        self.assertEqual(data.experimental_setup.ipd, 65.0)
        self.assertEqual(data.experimental_setup.d, 700.0)
    
    def test_set_experimental_setup_mixed_units(self):
        """Test set_experimental_setup with mixed unit formats"""
        data = pp.EyeData(left_x=[1,2], left_y=[3,4], left_pupil=[5,6], sampling_rate=1000)
        
        # Mix plain numbers, strings, and Quantities - eye frame for direct r
        data.set_experimental_setup(
            camera_spherical=("20 deg", "-90 deg", "600 mm"),
            eye_to_screen_center="0.7 m",
            physical_screen_size=(52 * pp.ureg.cm, "290 mm")
        )
        
        # All should be stored in mm
        self.assertEqual(data.experimental_setup.r, 600.0)
        self.assertEqual(data.experimental_setup.d, 700.0)
        self.assertEqual(data.experimental_setup.physical_screen_width, 520.0)
        self.assertEqual(data.experimental_setup.physical_screen_height, 290.0)


if __name__ == '__main__':
    unittest.main() 