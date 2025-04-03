import unittest
import sys
import numpy as np
sys.path.insert(0,"..")
import pypillometry as pp
from pypillometry.example_data import get_rlmw_002_short
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
        d = pp.EyeData(left_x=[1,2], left_y=[3,4])
        self.assertEqual(d.__class__, pp.EyeData)
        
        # Test with invalid data (missing y coordinate)
        with self.assertRaises(ValueError):
            pp.EyeData(left_x=[1,2])
            
        # Test with complete data
        self.assertEqual(self.eyedata.name, "test_data")
        self.assertEqual(self.eyedata.fs, 1000)
        self.assertEqual(self.eyedata.screen_width, 1920)
        self.assertEqual(self.eyedata.screen_height, 1080)
        self.assertEqual(self.eyedata.physical_screen_width, 53.34)
        self.assertEqual(self.eyedata.physical_screen_height, 30.0)
        self.assertEqual(self.eyedata.screen_eye_distance, 60.0)

    def test_summary(self):
        """Test the summary method"""
        summary = self.eyedata.summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('sampling_rate', summary)
        self.assertIn('duration', summary)
        self.assertIn('n_events', summary)

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
        
        # Test correction for single eye
        corrected_left = self.eyedata.correct_pupil_foreshortening(eyes=['left'])
        self.assertEqual(corrected_left.__class__, pp.EyeData)
        
        # Test with custom midpoint
        corrected_custom = self.eyedata.correct_pupil_foreshortening(
            eyes=['left'], 
            midpoint=(960, 540)  # Center of 1920x1080 screen
        )
        self.assertEqual(corrected_custom.__class__, pp.EyeData)
        
        # Test inplace modification
        original_pupil = self.eyedata.data['left_pupil'].copy()
        self.eyedata.correct_pupil_foreshortening(eyes=['left'], inplace=True)
        self.assertFalse(np.array_equal(original_pupil, self.eyedata.data['left_pupil']))

    def test_plot_property(self):
        """Test the plot property"""
        plotter = self.eyedata.plot
        self.assertEqual(plotter.__class__, pp.EyePlotter)
        self.assertEqual(plotter.data, self.eyedata)

if __name__ == '__main__':
    unittest.main() 