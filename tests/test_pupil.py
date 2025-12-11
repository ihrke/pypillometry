import unittest
import sys
import numpy as np
import pypillometry as pp
from pypillometry.signal.pupil import (
    pupil_kernel_t, 
    lowpass_filter_iterative,
    pupil_signal_quality
)

class TestPupil(unittest.TestCase):
    def setUp(self):
        self.d = pp.get_example_data("rlmw_002_short")
        
    def test_pupil_kernel_t(self):
        pupil_kernel_t([1,2], 10, 900)


class TestLowpassFilterIterative(unittest.TestCase):
    """Tests for lowpass_filter_iterative()"""
    
    def test_no_nan_matches_regular_filter(self):
        """With no NaN, should produce similar results to regular lowpass"""
        from pypillometry.signal.baseline import butter_lowpass_filter
        
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4*np.pi, 500)) + 0.1 * np.random.randn(500)
        fs = 100
        cutoff = 2.0
        
        result_iterative = lowpass_filter_iterative(signal, cutoff, fs)
        result_regular = butter_lowpass_filter(signal, cutoff, fs, order=2)
        
        # Should be very close (not exact due to different default order)
        np.testing.assert_allclose(result_iterative, result_regular, rtol=1e-10)
    
    def test_with_nan_returns_valid_output(self):
        """With NaN values, should return array without NaN"""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4*np.pi, 500)) + 0.1 * np.random.randn(500)
        signal[100:150] = np.nan  # Add gap
        
        result = lowpass_filter_iterative(signal, cutoff=2.0, fs=100)
        
        # Output should have no NaN
        self.assertFalse(np.any(np.isnan(result)))
        # Output should have same shape
        self.assertEqual(result.shape, signal.shape)
    
    def test_smooth_across_nan_gaps(self):
        """Output should be smooth across NaN gaps (no sharp jumps)"""
        # Create signal with a gap
        signal = np.ones(200) * 100  # Flat signal
        signal[80:120] = np.nan  # Gap in middle
        
        result = lowpass_filter_iterative(signal, cutoff=2.0, fs=100)
        
        # The filled gap should be close to 100 (the surrounding values)
        gap_values = result[80:120]
        np.testing.assert_allclose(gap_values, 100, atol=1)
    
    def test_all_nan_returns_copy(self):
        """All NaN input should return NaN array"""
        signal = np.full(100, np.nan)
        result = lowpass_filter_iterative(signal, cutoff=2.0, fs=100)
        
        self.assertTrue(np.all(np.isnan(result)))


class TestPupilSignalQuality(unittest.TestCase):
    """Tests for pupil_signal_quality()"""
    
    def setUp(self):
        """Create test signals"""
        np.random.seed(42)
        n = 1000
        fs = 100
        
        # Clean signal: slow sine wave
        t = np.linspace(0, 10, n)
        self.clean_signal = 500 + 50 * np.sin(2 * np.pi * 0.5 * t)
        
        # Noisy signal: same sine wave + high frequency noise
        self.noisy_signal = self.clean_signal + 20 * np.random.randn(n)
        
        # Mask (all valid)
        self.mask_none = np.zeros(n, dtype=bool)
        
        # Mask with some invalid
        self.mask_partial = np.zeros(n, dtype=bool)
        self.mask_partial[200:250] = True
        
        self.fs = fs
    
    def test_returns_correct_shape(self):
        """Output should have same shape as input"""
        for metric in ["snr", "snr_db", "noise_power", "noise_cv"]:
            result = pupil_signal_quality(
                self.noisy_signal, self.fs, 
                mask=self.mask_none, metric=metric
            )
            self.assertEqual(result.shape, self.noisy_signal.shape)
    
    def test_invalid_metric_raises_error(self):
        """Invalid metric should raise ValueError"""
        with self.assertRaises(ValueError):
            pupil_signal_quality(
                self.noisy_signal, self.fs,
                mask=self.mask_none, metric="invalid"
            )
    
    def test_missing_mask_raises_error(self):
        """Regular array without mask should raise ValueError"""
        with self.assertRaises(ValueError):
            pupil_signal_quality(self.noisy_signal, self.fs, metric="snr")
    
    def test_nan_in_valid_data_raises_error(self):
        """NaN in unmasked data should raise ValueError"""
        signal_with_nan = self.noisy_signal.copy()
        signal_with_nan[50] = np.nan  # NaN in valid region
        
        with self.assertRaises(ValueError):
            pupil_signal_quality(
                signal_with_nan, self.fs,
                mask=self.mask_none, metric="snr"
            )
    
    def test_masked_locations_are_zero(self):
        """Output should be 0 at masked locations"""
        result = pupil_signal_quality(
            self.noisy_signal, self.fs,
            mask=self.mask_partial, metric="snr"
        )
        
        masked_values = result[self.mask_partial]
        self.assertTrue(np.all(masked_values == 0))
    
    def test_masked_array_input(self):
        """Should accept numpy masked array"""
        masked_signal = np.ma.array(self.noisy_signal, mask=self.mask_partial)
        
        result = pupil_signal_quality(masked_signal, self.fs, metric="snr")
        
        self.assertEqual(result.shape, self.noisy_signal.shape)
        self.assertTrue(np.all(result[self.mask_partial] == 0))
    
    def test_snr_positive(self):
        """SNR should be positive for valid data"""
        result = pupil_signal_quality(
            self.noisy_signal, self.fs,
            mask=self.mask_none, metric="snr"
        )
        
        self.assertTrue(np.all(result > 0))
    
    def test_noise_cv_reasonable_range(self):
        """Noise CV should be in reasonable range (0-1 for typical data)"""
        result = pupil_signal_quality(
            self.noisy_signal, self.fs,
            mask=self.mask_none, metric="noise_cv"
        )
        
        # CV should be positive and typically < 1 for reasonable data
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.mean(result) < 1)


if __name__ == '__main__':
    unittest.main()