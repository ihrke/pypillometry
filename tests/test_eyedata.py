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
        merged = self.eyedata.blinks_merge(eyes=['left'], variables=['pupil'], distance=100, units="ms")
        blinks = merged.get_blinks('left', 'pupil')
        
        # blinks is now an Intervals object
        from pypillometry.intervals import Intervals
        self.assertIsInstance(blinks, Intervals)
        self.assertEqual(len(blinks), 1)  # Should merge into one blink
        
        # Convert to integer indices for indexing
        blinks_ix = blinks.as_index(merged)
        self.assertEqual(blinks_ix[0, 0], 100)  # Start of first blink
        self.assertEqual(blinks_ix[0, 1], 350)  # End of last blink

        # Test merging with distance specified in seconds
        merged_sec = self.eyedata.blinks_merge(eyes=['left'], variables=['pupil'], distance=0.1, units="sec")
        blinks_sec = merged_sec.get_blinks('left', 'pupil')
        blinks_ix_sec = blinks_sec.as_index(merged_sec)
        self.assertEqual(blinks_ix_sec[0, 0], 100)
        self.assertEqual(blinks_ix_sec[0, 1], 350)

        # Ensure blinks beyond distance remain separate
        self.eyedata.set_blinks('left', 'pupil', np.array([[100, 200], [250, 350]]))
        separated = self.eyedata.blinks_merge(eyes=['left'], variables=['pupil'], distance=10, units="ms")
        blinks_sep = separated.get_blinks('left', 'pupil')
        blinks_ix_sep = blinks_sep.as_index(separated)
        self.assertEqual(len(blinks_ix_sep), 2)
        np.testing.assert_array_equal(blinks_ix_sep[0], [100, 200])
        np.testing.assert_array_equal(blinks_ix_sep[1], [250, 350])

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
        blink_indices = blinks.as_index(result)
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
        self.eyedata.set_blinks('left', 'pupil', np.array([[100, 200], [250, 350]]))
        
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
        self.assertEqual(len(blinks_sec), 2)
    
    def test_get_blinks_merged_multiple_eyes(self):
        """Test merging blinks across multiple eyes"""
        from pypillometry.intervals import Intervals
        
        # Create blinks for both eyes
        self.eyedata.set_blinks('left', 'pupil', np.array([[100, 200]]))
        self.eyedata.set_blinks('right', 'pupil', np.array([[150, 250]]))
        
        # Get merged blinks
        blinks = self.eyedata.get_blinks(['left', 'right'], 'pupil')
        
        self.assertIsInstance(blinks, Intervals)
        # Should merge overlapping blinks
        self.assertGreater(len(blinks), 0)

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


if __name__ == '__main__':
    unittest.main() 