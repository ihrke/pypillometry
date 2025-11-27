import unittest
import sys
import numpy as np
sys.path.insert(0,"..")
import pypillometry as pp
from pypillometry.eyedata import ExperimentalSetup
import pytest

class TestGazeData(unittest.TestCase):
    def setUp(self):
        #self.d=pp.GazeData.from_file("data/test.pd")
        pass
    def test_from_file(self):
        d = pp.get_example_data("rlmw_002_short")
        print(d)
    def test_scale(self):
        d = pp.GazeData(sampling_rate=10, left_x=[1,2,1,2], left_y=[3,4,3,4])
        d=d.scale(["x",'y'])
        self.assertEqual(d.data["left","x"].mean(), 0)
        self.assertEqual(d.data["left","y"].mean(), 0)
        self.assertEqual(d.params["scale"]["mean"]["left"]["x"], 1.5)
        self.assertEqual(d.params["scale"]["mean"]["left"]["y"], 3.5)
    def test_scale_fixed(self):
        d = pp.GazeData(sampling_rate=10, left_x=[1,2,1,2], left_y=[3,4,3,4])
        scalepars = {
            "left": {
                "x": 1,
                "y": 3
            }
        }
        sdpars = {
            "left": {
                "x": 0.5,
                "y": 0.5
            }
        }
        d=d.scale(["x",'y'], mean=scalepars, sd=sdpars)
        self.assertEqual(d.data["left","x"].mean(), 1)
        self.assertEqual(d.data["left","y"].mean(), 1)
    def test_unscale(self):
        d = pp.GazeData(sampling_rate=10, left_x=[1,2,1,2], left_y=[3,4,3,4])
        d=d.scale(["x",'y'])
        d=d.unscale(["x",'y'])
        self.assertEqual(d.data["left","x"].mean(), 1.5)
        self.assertEqual(d.data["left","y"].mean(), 3.5)
    
    def test_scale_preserves_mask(self):
        """Test that scale() preserves existing masks"""
        d = pp.GazeData(sampling_rate=10, left_x=[1, 2, 3, 4], left_y=[5, 6, 7, 8])
        
        # Manually mask some points
        d.data.mask['left_x'][1] = 1
        d.data.mask['left_y'][2] = 1
        
        # Scale the data
        d_scaled = d.scale(['x', 'y'], inplace=False)
        
        # Verify masks are preserved
        self.assertEqual(d_scaled.data.mask['left_x'][1], 1)
        self.assertEqual(d_scaled.data.mask['left_y'][2], 1)
        
        # Verify other points are not masked
        self.assertEqual(d_scaled.data.mask['left_x'][0], 0)
        self.assertEqual(d_scaled.data.mask['left_y'][0], 0)
    
    def test_unscale_preserves_mask(self):
        """Test that unscale() preserves existing masks"""
        d = pp.GazeData(sampling_rate=10, left_x=[1, 2, 3, 4], left_y=[5, 6, 7, 8])
        
        # Manually mask some points
        d.data.mask['left_x'][1] = 1
        d.data.mask['left_y'][2] = 1
        
        # Scale and then unscale
        d_scaled = d.scale(['x', 'y'], inplace=False)
        d_unscaled = d_scaled.unscale(['x', 'y'], inplace=False)
        
        # Verify masks are preserved through both operations
        self.assertEqual(d_unscaled.data.mask['left_x'][1], 1)
        self.assertEqual(d_unscaled.data.mask['left_y'][2], 1)
        
        # Verify other points are not masked
        self.assertEqual(d_unscaled.data.mask['left_x'][0], 0)
        self.assertEqual(d_unscaled.data.mask['left_y'][0], 0)
    
    def test_mask_eye_divergences_percentile(self):
        """Test masking eye divergences with percentile threshold"""
        # Create data with both eyes where some points have large divergence
        left_x = np.array([100, 100, 100, 100, 100, 100, 100, 100, 200, 100])
        left_y = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200, 200])
        right_x = np.array([105, 105, 105, 105, 105, 105, 105, 105, 105, 105])
        right_y = np.array([205, 205, 205, 205, 205, 205, 205, 205, 205, 205])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x, left_y=left_y,
            right_x=right_x, right_y=right_y
        )
        
        # Store original mask state
        orig_left_x_mask = d.data.mask['left_x'].copy()
        
        # Apply divergence masking at 90th percentile with distance stored
        d_masked = d.mask_eye_divergences(threshold=0.9, thr_type="percentile", store_as="dist", inplace=False)
        
        # Verify distance was stored
        self.assertIn('dist', d_masked.data)
        
        # Calculate expected distance
        expected_dist = np.sqrt((left_x - right_x)**2 + (left_y - right_y)**2)
        
        # Verify distance calculation
        np.testing.assert_allclose(
            d_masked.data['dist'],
            expected_dist,
            rtol=1e-5
        )
        
        # Verify that the point with large divergence (index 8) is masked
        self.assertEqual(d_masked.data.mask['left_x'][8], 1)
        self.assertEqual(d_masked.data.mask['left_y'][8], 1)
        self.assertEqual(d_masked.data.mask['right_x'][8], 1)
        self.assertEqual(d_masked.data.mask['right_y'][8], 1)
        
        # Verify original data is not modified (inplace=False)
        np.testing.assert_array_equal(d.data.mask['left_x'], orig_left_x_mask)
    
    def test_mask_eye_divergences_pixel(self):
        """Test masking eye divergences with pixel threshold"""
        # Create data with known distances
        left_x = np.array([100, 100, 100, 100, 100])
        left_y = np.array([200, 200, 200, 200, 200])
        right_x = np.array([105, 110, 115, 120, 125])  # Increasing distance
        right_y = np.array([205, 210, 215, 220, 225])  # Increasing distance
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x, left_y=left_y,
            right_x=right_x, right_y=right_y
        )
        
        # Apply divergence masking at 10 pixels threshold
        d_masked = d.mask_eye_divergences(threshold=10, thr_type="pixel", store_as=None, inplace=False)
        
        # Verify distance was NOT stored (store_as=None)
        self.assertNotIn('dist', d_masked.data)
        
        # Calculate distances
        distances = np.sqrt((left_x - right_x)**2 + (left_y - right_y)**2)
        
        # Points with distance > 10 should be masked
        for i, dist in enumerate(distances):
            if dist > 10:
                self.assertEqual(d_masked.data.mask['left_x'][i], 1, 
                                f"Point {i} with distance {dist:.2f} should be masked")
            else:
                self.assertEqual(d_masked.data.mask['left_x'][i], 0,
                                f"Point {i} with distance {dist:.2f} should not be masked")
    
    def test_mask_eye_divergences_inplace(self):
        """Test that inplace parameter works correctly"""
        left_x = np.array([100, 100, 200, 100])
        left_y = np.array([200, 200, 200, 200])
        right_x = np.array([105, 105, 105, 105])
        right_y = np.array([205, 205, 205, 205])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x, left_y=left_y,
            right_x=right_x, right_y=right_y
        )
        
        # Apply in-place
        result = d.mask_eye_divergences(threshold=0.9, thr_type="percentile", store_as="mydist", inplace=True)
        
        # Result should be the same object
        self.assertIs(result, d)
        
        # Verify distance was stored in original object
        self.assertIn('mydist', d.data)
    
    def test_mask_eye_divergences_requires_both_eyes(self):
        """Test that function raises error if both eyes are not available"""
        # Create data with only left eye
        d = pp.GazeData(
            sampling_rate=10,
            left_x=[100, 100, 100],
            left_y=[200, 200, 200]
        )
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as cm:
            d.mask_eye_divergences(threshold=0.9, thr_type="percentile")
        
        self.assertIn("Both left and right eye data are required", str(cm.exception))
    
    def test_mask_eye_divergences_invalid_thr_type(self):
        """Test that invalid thr_type raises error"""
        d = pp.GazeData(
            sampling_rate=10,
            left_x=[100, 100], left_y=[200, 200],
            right_x=[105, 105], right_y=[205, 205]
        )
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as cm:
            d.mask_eye_divergences(threshold=0.9, thr_type="invalid")
        
        self.assertIn("thr_type must be 'percentile' or 'pixel'", str(cm.exception))
    
    def test_mask_eye_divergences_percentile_out_of_range(self):
        """Test that percentile threshold must be between 0 and 1"""
        d = pp.GazeData(
            sampling_rate=10,
            left_x=[100, 100], left_y=[200, 200],
            right_x=[105, 105], right_y=[205, 205]
        )
        
        # Should raise ValueError for threshold > 1
        with self.assertRaises(ValueError) as cm:
            d.mask_eye_divergences(threshold=99, thr_type="percentile")
        
        self.assertIn("threshold must be between 0 and 1", str(cm.exception))
    
    def test_mask_eye_divergences_preserves_existing_mask(self):
        """Test that existing masks are preserved and combined with divergence mask"""
        left_x = np.array([100, 100, 100, 100, 100])
        left_y = np.array([200, 200, 200, 200, 200])
        right_x = np.array([105, 105, 105, 105, 105])
        right_y = np.array([205, 205, 205, 205, 205])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x, left_y=left_y,
            right_x=right_x, right_y=right_y
        )
        
        # Manually mask some points before divergence masking
        d.data.mask['left_x'][0] = 1
        d.data.mask['left_y'][0] = 1
        
        # Apply divergence masking (with very low threshold to mask nothing new)
        d_masked = d.mask_eye_divergences(threshold=1000, thr_type="pixel", inplace=False)
        
        # Original manual mask should still be there
        self.assertEqual(d_masked.data.mask['left_x'][0], 1)
        self.assertEqual(d_masked.data.mask['left_y'][0], 1)
    
    def test_mask_eye_divergences_with_masked_input(self):
        """Test that function works correctly with pre-existing masks"""
        left_x = np.array([100, 100, 100, 100])
        left_y = np.array([200, 200, 200, 200])
        right_x = np.array([105, 105, 105, 105])
        right_y = np.array([205, 205, 205, 205])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x, left_y=left_y,
            right_x=right_x, right_y=right_y
        )
        
        # Manually add mask to simulate pre-existing masked data
        d.data.mask['left_x'][1] = 1
        d.data.mask['left_y'][1] = 1
        
        # Apply divergence masking
        d_masked = d.mask_eye_divergences(threshold=0.5, thr_type="percentile", store_as="dist", inplace=False)
        
        # Original masked point should still be masked
        self.assertEqual(d_masked.data.mask['left_x'][1], 1)
        
        # Distance array should also have the masked point
        self.assertEqual(d_masked.data.mask['dist'][1], 1)
    
    def test_mask_eye_divergences_apply_mask_false(self):
        """Test mask_eye_divergences with apply_mask=False returns Intervals"""
        from pypillometry.intervals import Intervals
        
        # Create test data with divergence
        d = pp.GazeData(
            left_x=np.array([100, 110, 120, 130, 140]),
            left_y=np.array([100, 100, 100, 100, 100]),
            right_x=np.array([100, 110, 500, 130, 140]),  # Large divergence at index 2
            right_y=np.array([100, 100, 100, 100, 100]),
            sampling_rate=1000
        )
        
        # Detect divergences without applying mask
        result = d.mask_eye_divergences(threshold=0.5, thr_type="percentile", 
                                       apply_mask=False, inplace=False)
        
        # Should return a single Intervals object
        self.assertIsInstance(result, Intervals)
        self.assertEqual(result.label, "eye_divergences")
        
        # Should have detected divergence intervals
        self.assertGreater(len(result), 0)
        
        # Mask should NOT be applied to original data
        original_mask_sum = np.sum(d.data.mask['left_x'])
        self.assertEqual(original_mask_sum, 0)
    
    def test_mask_eye_divergences_apply_mask_true_default(self):
        """Test that apply_mask=True is the default behavior"""
        d = pp.GazeData(
            left_x=np.array([100, 110, 120, 130, 140]),
            left_y=np.array([100, 100, 100, 100, 100]),
            right_x=np.array([100, 110, 500, 130, 140]),
            right_y=np.array([100, 100, 100, 100, 100]),
            sampling_rate=1000
        )
        
        # Call without apply_mask argument (should default to True)
        result = d.mask_eye_divergences(threshold=0.5, thr_type="percentile", 
                                       inplace=False)
        
        # Should return GazeData object (self)
        self.assertIsInstance(result, pp.GazeData)
        
        # Mask should be applied
        mask_sum = np.sum(result.data.mask['left_x'])
        self.assertGreater(mask_sum, 0)


class TestMaskOffscreenCoords(unittest.TestCase):
    """Test suite for mask_offscreen_coords function"""
    
    def test_mask_offscreen_coords_basic(self):
        """Test basic masking of offscreen coordinates"""
        # Create data where some points are outside screen boundaries
        left_x = np.array([100, 200, 300, 1500, 500])  # 1500 is outside
        left_y = np.array([100, 200, 300, 400, 1200])  # 1200 is outside
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x,
            left_y=left_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Store original mask state
        orig_mask = d.data.mask['left_x'].copy()
        
        # Apply offscreen masking
        d_masked = d.mask_offscreen_coords(eyes='left', inplace=False)
        
        # Verify that offscreen points are masked
        self.assertEqual(d_masked.data.mask['left_x'][3], 1)  # x=1500 > 1280
        self.assertEqual(d_masked.data.mask['left_y'][4], 1)  # y=1200 > 1024
        
        # Verify that onscreen points are not masked
        self.assertEqual(d_masked.data.mask['left_x'][0], 0)
        self.assertEqual(d_masked.data.mask['left_x'][1], 0)
        
        # Verify original data is not modified (inplace=False)
        np.testing.assert_array_equal(d.data.mask['left_x'], orig_mask)
    
    def test_mask_offscreen_coords_both_eyes(self):
        """Test masking offscreen coordinates for both eyes"""
        left_x = np.array([100, 200, 1500, 400])
        left_y = np.array([100, 200, 300, 400])
        right_x = np.array([100, 200, 300, 1500])
        right_y = np.array([100, 1200, 300, 400])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x, left_y=left_y,
            right_x=right_x, right_y=right_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Mask both eyes
        d_masked = d.mask_offscreen_coords(inplace=False)
        
        # Verify left eye masking
        self.assertEqual(d_masked.data.mask['left_x'][2], 1)  # left_x=1500 offscreen
        
        # Verify right eye masking
        self.assertEqual(d_masked.data.mask['right_y'][1], 1)  # right_y=1200 offscreen
        self.assertEqual(d_masked.data.mask['right_x'][3], 1)  # right_x=1500 offscreen
    
    def test_mask_offscreen_coords_apply_mask_false(self):
        """Test mask_offscreen_coords with apply_mask=False returns dict of Intervals"""
        from pypillometry.intervals import Intervals
        
        # Create data with offscreen points
        d = pp.GazeData(
            left_x=np.array([100, 1500, 300, 400, 1600]),
            left_y=np.array([100, 200, 300, 400, 500]),
            sampling_rate=1000,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Detect offscreen without applying mask
        result = d.mask_offscreen_coords(apply_mask=False, inplace=False)
        
        # Should return a dict of Intervals objects
        self.assertIsInstance(result, dict)
        self.assertIn('left', result)
        self.assertIsInstance(result['left'], Intervals)
        self.assertEqual(result['left'].label, "left_offscreen")
        
        # Should have detected offscreen intervals
        self.assertGreater(len(result['left']), 0)
        
        # Mask should NOT be applied to original data
        original_mask_sum = np.sum(d.data.mask['left_x'])
        self.assertEqual(original_mask_sum, 0)
    
    def test_mask_offscreen_coords_apply_mask_true_default(self):
        """Test that apply_mask=True is the default behavior"""
        d = pp.GazeData(
            left_x=np.array([100, 1500, 300, 400]),
            left_y=np.array([100, 200, 300, 400]),
            sampling_rate=1000,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Call without apply_mask argument (should default to True)
        result = d.mask_offscreen_coords(eyes='left', inplace=False)
        
        # Should return GazeData object
        self.assertIsInstance(result, pp.GazeData)
        
        # Mask should be applied
        mask_sum = np.sum(result.data.mask['left_x'])
        self.assertGreater(mask_sum, 0)
    
    def test_mask_offscreen_coords_inplace(self):
        """Test that inplace parameter works correctly"""
        left_x = np.array([100, 1500, 300, 400])
        left_y = np.array([100, 200, 300, 400])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x,
            left_y=left_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Apply in-place
        result = d.mask_offscreen_coords(eyes='left', inplace=True)
        
        # Result should be the same object
        self.assertIs(result, d)
        
        # Verify mask was applied to original object
        self.assertEqual(d.data.mask['left_x'][1], 1)
    
    def test_mask_offscreen_coords_no_screen_limits(self):
        """Test that function raises error if screen limits are not set"""
        # Create data without setting screen resolution
        d = pp.GazeData(
            sampling_rate=10,
            left_x=[100, 200, 300],
            left_y=[100, 200, 300]
        )
        
        # Should raise ValueError when experimental_setup is not set
        with self.assertRaises(ValueError) as cm:
            d.mask_offscreen_coords(eyes='left')
        
        self.assertIn("experimental_setup not set", str(cm.exception))
    
    def test_mask_offscreen_coords_no_offscreen_data(self):
        """Test function behavior when no coordinates are offscreen"""
        left_x = np.array([100, 200, 300, 400])
        left_y = np.array([100, 200, 300, 400])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x,
            left_y=left_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Apply offscreen masking
        d_masked = d.mask_offscreen_coords(eyes='left', inplace=False)
        
        # No points should be masked
        self.assertEqual(np.sum(d_masked.data.mask['left_x']), 0)
        self.assertEqual(np.sum(d_masked.data.mask['left_y']), 0)
    
    def test_mask_offscreen_coords_negative_coords(self):
        """Test masking of negative coordinates (offscreen below/left of origin)"""
        left_x = np.array([-10, 100, 200, 300])
        left_y = np.array([100, -5, 200, 300])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x,
            left_y=left_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Apply offscreen masking
        d_masked = d.mask_offscreen_coords(eyes='left', inplace=False)
        
        # Verify negative coordinates are masked
        self.assertEqual(d_masked.data.mask['left_x'][0], 1)  # x=-10 < 0
        self.assertEqual(d_masked.data.mask['left_y'][1], 1)  # y=-5 < 0
    
    def test_mask_offscreen_coords_ignore_existing_mask_false(self):
        """Test that existing masks are respected when ignore_existing_mask=False"""
        left_x = np.array([100, 1500, 300, 1600, 500])
        left_y = np.array([100, 200, 300, 400, 500])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x,
            left_y=left_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Manually mask index 1 (which is offscreen)
        d.data.mask['left_x'][1] = 1
        d.data.mask['left_y'][1] = 1
        
        # Apply offscreen masking with ignore_existing_mask=False (default)
        result = d.mask_offscreen_coords(eyes='left', ignore_existing_mask=False, 
                                        apply_mask=False, inplace=False)
        
        # Index 3 should be detected as offscreen (x=1600)
        # Index 1 should NOT be in the intervals since it was already masked
        intervals = result['left'].intervals
        
        # Should detect index 3 as offscreen
        offscreen_indices = [idx for start, end in intervals for idx in range(start, end)]
        self.assertIn(3, offscreen_indices)
    
    def test_mask_offscreen_coords_ignore_existing_mask_true(self):
        """Test that existing masks are ignored when ignore_existing_mask=True"""
        left_x = np.array([100, 1500, 300, 1600, 500])
        left_y = np.array([100, 200, 300, 400, 500])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x,
            left_y=left_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Manually mask index 1 (which is also offscreen)
        d.data.mask['left_x'][1] = 1
        d.data.mask['left_y'][1] = 1
        
        # Apply offscreen masking with ignore_existing_mask=True
        result = d.mask_offscreen_coords(eyes='left', ignore_existing_mask=True, 
                                        apply_mask=False, inplace=False)
        
        # Both index 1 and 3 should be detected as offscreen
        intervals = result['left'].intervals
        offscreen_indices = [idx for start, end in intervals for idx in range(start, end)]
        
        self.assertIn(1, offscreen_indices)  # Previously masked but still detected
        self.assertIn(3, offscreen_indices)
    
    def test_mask_offscreen_coords_preserves_existing_mask(self):
        """Test that existing masks are preserved and combined with offscreen mask"""
        left_x = np.array([100, 200, 300, 400, 500])
        left_y = np.array([100, 200, 300, 400, 500])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x,
            left_y=left_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Manually mask some points before offscreen masking
        d.data.mask['left_x'][0] = 1
        d.data.mask['left_y'][0] = 1
        
        # Apply offscreen masking (no points should be offscreen)
        d_masked = d.mask_offscreen_coords(eyes='left', inplace=False)
        
        # Original manual mask should still be there
        self.assertEqual(d_masked.data.mask['left_x'][0], 1)
        self.assertEqual(d_masked.data.mask['left_y'][0], 1)
    
    def test_mask_offscreen_coords_specific_eye(self):
        """Test masking offscreen coordinates for a specific eye only"""
        left_x = np.array([100, 1500, 300, 400])
        left_y = np.array([100, 200, 300, 400])
        right_x = np.array([1500, 200, 300, 400])
        right_y = np.array([100, 200, 300, 400])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x, left_y=left_y,
            right_x=right_x, right_y=right_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Mask only left eye
        d_masked = d.mask_offscreen_coords(eyes='left', inplace=False)
        
        # Left eye offscreen point should be masked
        self.assertEqual(d_masked.data.mask['left_x'][1], 1)
        
        # Right eye offscreen point should NOT be masked
        self.assertEqual(d_masked.data.mask['right_x'][0], 0)
    
    def test_mask_offscreen_coords_multiple_eyes_list(self):
        """Test masking offscreen coordinates for multiple eyes using list"""
        left_x = np.array([100, 1500, 300, 400])
        left_y = np.array([100, 200, 300, 400])
        right_x = np.array([1500, 200, 300, 400])
        right_y = np.array([100, 200, 300, 400])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x, left_y=left_y,
            right_x=right_x, right_y=right_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Mask both eyes using list
        d_masked = d.mask_offscreen_coords(eyes=['left', 'right'], inplace=False)
        
        # Both eyes' offscreen points should be masked
        self.assertEqual(d_masked.data.mask['left_x'][1], 1)
        self.assertEqual(d_masked.data.mask['right_x'][0], 1)
    
    def test_mask_offscreen_coords_empty_eyes_parameter(self):
        """Test that empty eyes parameter defaults to all available eyes"""
        left_x = np.array([100, 1500, 300, 400])
        left_y = np.array([100, 200, 300, 400])
        right_x = np.array([1500, 200, 300, 400])
        right_y = np.array([100, 200, 300, 400])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x, left_y=left_y,
            right_x=right_x, right_y=right_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Call with empty eyes parameter
        d_masked = d.mask_offscreen_coords(eyes=[], inplace=False)
        
        # Both eyes should be processed
        self.assertEqual(d_masked.data.mask['left_x'][1], 1)
        self.assertEqual(d_masked.data.mask['right_x'][0], 1)
    
    def test_mask_offscreen_coords_chaining(self):
        """Test that mask_offscreen_coords can be chained with other operations"""
        left_x = np.array([100, 1500, 300, 1600])
        left_y = np.array([100, 200, 1200, 400])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x,
            left_y=left_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        # Chain operations
        d_processed = (d
            .mask_offscreen_coords(eyes='left', inplace=False)
            .scale(['x', 'y'])
        )
        
        # Verify masks were applied and preserved through chaining
        self.assertEqual(d_processed.data.mask['left_x'][1], 1)
        self.assertEqual(d_processed.data.mask['left_y'][2], 1)
        
        # Verify scaling was applied
        self.assertIn('scale', d_processed.params)
    
    def test_mask_offscreen_coords_at_boundary(self):
        """Test that coordinates exactly at screen boundaries are not masked"""
        # Test coordinates at exact boundaries
        left_x = np.array([0, 640, 1280, 1281])
        left_y = np.array([0, 512, 1024, 1025])
        
        d = pp.GazeData(
            sampling_rate=10,
            left_x=left_x,
            left_y=left_y,
            experimental_setup=ExperimentalSetup(screen_resolution=(1280, 1024))
        )
        
        d_masked = d.mask_offscreen_coords(eyes='left', inplace=False)
        
        # Coordinates at 0, 1280, 1024 should NOT be masked (at boundary)
        self.assertEqual(d_masked.data.mask['left_x'][0], 0)  # x=0
        self.assertEqual(d_masked.data.mask['left_x'][2], 0)  # x=1280
        self.assertEqual(d_masked.data.mask['left_y'][2], 0)  # y=1024
        
        # Coordinates beyond boundary should be masked
        self.assertEqual(d_masked.data.mask['left_x'][3], 1)  # x=1281 > 1280
        self.assertEqual(d_masked.data.mask['left_y'][3], 1)  # y=1025 > 1024

if __name__ == '__main__':
    unittest.main()