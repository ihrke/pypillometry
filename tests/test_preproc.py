import unittest
import numpy as np
from pypillometry.signal.preproc import smooth_window, detect_blinks_velocity


class TestSmoothWindow(unittest.TestCase):
    """Tests for smooth_window() function"""
    
    def setUp(self):
        """Create test signals"""
        np.random.seed(42)
        # Simple ramp signal
        self.ramp = np.arange(100, dtype=float)
        # Noisy sine wave
        t = np.linspace(0, 4*np.pi, 200)
        self.sine = np.sin(t) + 0.1 * np.random.randn(200)
    
    def test_center_direction_default(self):
        """Center direction should be the default"""
        result1 = smooth_window(self.ramp, window_len=5)
        result2 = smooth_window(self.ramp, window_len=5, direction='center')
        np.testing.assert_array_equal(result1, result2)
    
    def test_output_same_length(self):
        """Output should have same length as input for all directions"""
        for direction in ['center', 'backward', 'forward']:
            result = smooth_window(self.ramp, window_len=11, direction=direction)
            self.assertEqual(len(result), len(self.ramp))
    
    def test_backward_uses_only_past(self):
        """Backward smoothing should only use past samples"""
        # Create signal with a step change
        signal = np.concatenate([np.zeros(50), np.ones(50)])
        result = smooth_window(signal, window_len=11, window='flat', direction='backward')
        
        # At index 50 (start of step), backward smooth should still be ~0
        # because it only looks at past samples (which are all 0)
        self.assertLess(result[50], 0.2)
        
        # At index 60 (10 samples into step), should be close to 1
        # because all 11 samples in window are now 1
        self.assertGreater(result[60], 0.9)
    
    def test_forward_uses_only_future(self):
        """Forward smoothing should only use future samples"""
        # Create signal with a step change
        signal = np.concatenate([np.zeros(50), np.ones(50)])
        result = smooth_window(signal, window_len=11, window='flat', direction='forward')
        
        # At index 49 (just before step), forward smooth should see the step
        self.assertGreater(result[49], 0.8)
        
        # At index 40 (10 samples before step), should still be ~0
        self.assertLess(result[40], 0.2)
    
    def test_center_symmetric(self):
        """Center smoothing should be symmetric"""
        signal = np.concatenate([np.zeros(50), np.ones(50)])
        result = smooth_window(signal, window_len=11, window='flat', direction='center')
        
        # At the step (index 50), center smooth should be ~0.5
        self.assertGreater(result[50], 0.3)
        self.assertLess(result[50], 0.7)
    
    def test_nan_raises_error(self):
        """NaN in signal should raise ValueError"""
        signal_with_nan = self.ramp.copy()
        signal_with_nan[50] = np.nan
        
        with self.assertRaises(ValueError) as ctx:
            smooth_window(signal_with_nan, window_len=5)
        
        self.assertIn("NaN", str(ctx.exception))
    
    def test_invalid_direction_raises_error(self):
        """Invalid direction should raise ValueError"""
        with self.assertRaises(ValueError):
            smooth_window(self.ramp, window_len=5, direction='invalid')
    
    def test_invalid_window_raises_error(self):
        """Invalid window type should raise ValueError"""
        with self.assertRaises(ValueError):
            smooth_window(self.ramp, window_len=5, window='invalid')
    
    def test_signal_too_short_raises_error(self):
        """Signal shorter than window should raise ValueError"""
        short_signal = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            smooth_window(short_signal, window_len=5)
    
    def test_all_window_types(self):
        """All window types should work"""
        for window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            result = smooth_window(self.sine, window_len=11, window=window)
            self.assertEqual(len(result), len(self.sine))
            # Smoothed signal should have lower variance than original
            self.assertLess(np.var(result), np.var(self.sine))
    
    def test_smoothing_reduces_noise(self):
        """Smoothing should reduce high-frequency noise"""
        for direction in ['center', 'backward', 'forward']:
            result = smooth_window(self.sine, window_len=11, direction=direction)
            # Smoothed signal should have lower variance
            self.assertLess(np.var(result), np.var(self.sine))


class TestDetectBlinksVelocity(unittest.TestCase):
    """Tests for detect_blinks_velocity() with asymmetric smoothing"""
    
    def test_handles_nan_without_crashing(self):
        """Should handle NaN values without crashing"""
        # Create signal with NaN gaps (typical eye tracker output)
        signal = np.ones(500) * 1000
        signal[100:150] = np.nan  # Missing data
        
        # Should not raise an error
        blinks = detect_blinks_velocity(signal, smooth_winsize=5, 
                                        vel_onset=-50, vel_offset=50,
                                        min_onset_len=1, min_offset_len=1)
        
        # Should return an array (may be empty for this signal)
        self.assertIsInstance(blinks, np.ndarray)
    
    def test_detects_velocity_based_blinks(self):
        """Should detect blinks based on velocity changes before/after invalid regions"""
        # Create signal with two blinks to avoid edge case in filtering logic
        signal = np.ones(500) * 1000
        
        # First blink
        signal[90:100] = np.linspace(1000, 100, 10)  # gradual drop
        signal[100:120] = 0  # blink
        signal[120:130] = np.linspace(100, 1000, 10)  # gradual rise
        
        # Second blink (needed to avoid edge case in onset detection)
        signal[200:210] = np.linspace(1000, 100, 10)  # gradual drop
        signal[210:230] = 0  # blink
        signal[230:240] = np.linspace(100, 1000, 10)  # gradual rise
        
        blinks = detect_blinks_velocity(signal, smooth_winsize=5, 
                                        vel_onset=-50, vel_offset=50,
                                        min_onset_len=2, min_offset_len=2)
        
        # Should detect at least one blink based on velocity changes
        self.assertGreaterEqual(len(blinks), 1)
    
    def test_returns_empty_for_flat_signal(self):
        """Flat signal should have no blinks"""
        signal = np.ones(200) * 100
        blinks = detect_blinks_velocity(signal, smooth_winsize=5,
                                        vel_onset=-10, vel_offset=10)
        self.assertEqual(len(blinks), 0)
    
    def test_blink_indices_valid(self):
        """Blink onset should be before offset"""
        signal = np.ones(200) * 100
        signal[50:60] = 0  # Blink
        
        blinks = detect_blinks_velocity(signal, smooth_winsize=5,
                                        vel_onset=-5, vel_offset=5,
                                        min_onset_len=1, min_offset_len=1)
        
        for onset, offset in blinks:
            self.assertLess(onset, offset)
            self.assertGreaterEqual(onset, 0)
            self.assertLess(offset, len(signal))


if __name__ == '__main__':
    unittest.main()
