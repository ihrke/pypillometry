import unittest
import sys
import numpy as np
sys.path.insert(0,"..")
import pypillometry as pp
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

if __name__ == '__main__':
    unittest.main()