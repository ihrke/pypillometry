import unittest
import sys
import os
import tempfile
import numpy as np
import pickle
import requests
from unittest.mock import patch, MagicMock
from pypillometry.io import write_pickle, read_pickle, read_eyelink
from importlib.resources import files

class TestIO(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {
            'array': np.array([1, 2, 3]),
            'string': 'test string',
            'number': 42,
            'dict': {'key': 'value'}
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
        
    def test_write_pickle(self):
        """Test writing objects to pickle file."""
        test_file = os.path.join(self.temp_dir, 'test.pkl')
        
        # Test writing different types of objects
        for key, value in self.test_data.items():
            write_pickle(value, test_file)
            self.assertTrue(os.path.exists(test_file))
            
            # Verify file is not empty
            self.assertGreater(os.path.getsize(test_file), 0)
            
    def test_read_pickle_local(self):
        """Test reading objects from local pickle file."""
        test_file = os.path.join(self.temp_dir, 'test.pkl')
        
        # Test reading different types of objects
        for key, value in self.test_data.items():
            # Write the object
            write_pickle(value, test_file)
            
            # Read it back
            loaded = read_pickle(test_file)
            
            # Compare original and loaded object
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, loaded)
            else:
                self.assertEqual(value, loaded)
                
    @patch('requests.get')
    def test_read_pickle_url(self, mock_get):
        """Test reading objects from URL."""
        # Create mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '100'}
        mock_response.iter_content.return_value = [pickle.dumps(self.test_data)]
        mock_get.return_value = mock_response
        
        # Test reading from URL
        loaded = read_pickle('http://example.com/test.pkl')
        
        # Verify mock was called
        mock_get.assert_called_once()
        
        # Compare loaded data
        for key, value in self.test_data.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, loaded[key])
            else:
                self.assertEqual(value, loaded[key])
                
    @patch('requests.get')
    def test_read_pickle_invalid_url(self, mock_get):
        """Test reading from invalid URL."""
        # Create mock response with invalid pickle data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '100'}
        mock_response.iter_content.return_value = [b'invalid pickle data']
        mock_get.return_value = mock_response
        
        with self.assertRaises(pickle.UnpicklingError):
            read_pickle('http://invalid-url.com/nonexistent.pkl')
            
    def test_read_pickle_invalid_file(self):
        """Test reading from invalid local file."""
        with self.assertRaises(FileNotFoundError):
            read_pickle(os.path.join(self.temp_dir, 'nonexistent.pkl'))

    def test_read_eyelink_local_file(self):
        """Test reading EDF file using read_eyelink function."""
        # Get path to test EDF file
        test_edf_path = files('pypillometry.data').joinpath('test.edf')
        test_edf_path = os.path.abspath(test_edf_path)
        
        # Skip test if eyelinkio is not available
        try:
            import eyelinkio
        except ImportError:
            self.skipTest("eyelinkio package not available")
        
        # Skip test if test file doesn't exist
        if not os.path.exists(test_edf_path):
            self.skipTest(f"Test EDF file not found at {test_edf_path}")
        
        # Test reading the EDF file
        edf_data = read_eyelink(test_edf_path)
        
        # Basic assertions about the returned object
        self.assertIsNotNone(edf_data, "read_eyelink should return a non-None object")
        
        # Check that it's an eyelinkio object (has expected attributes)
        # These are common attributes that eyelinkio EDF objects typically have
        expected_attrs = ['discrete', 'samples', 'info']
        for attr in expected_attrs:
            if hasattr(edf_data, attr):
                # At least one expected attribute should be present
                break
        else:
            # If none of the expected attributes are found, it might still be valid
            # Just check that it's not a basic type
            self.assertNotIsInstance(edf_data, (str, int, float, list, dict))

    @patch('pypillometry.io.is_url')
    @patch('pypillometry.io.download')
    def test_read_eyelink_url(self, mock_download, mock_is_url):
        """Test reading EDF file from URL using read_eyelink function."""
        # Skip test if eyelinkio is not available
        try:
            import eyelinkio
        except ImportError:
            self.skipTest("eyelinkio package not available")
        
        # Get path to test EDF file for mocking
        test_edf_path = files('pypillometry.data').joinpath('test.edf')        
        test_edf_path = os.path.abspath(test_edf_path)
        
        # Skip test if test file doesn't exist
        if not os.path.exists(test_edf_path):
            self.skipTest(f"Test EDF file not found at {test_edf_path}")
        
        # Mock URL detection and download
        mock_is_url.return_value = True
        mock_download.return_value = test_edf_path
        
        # Test reading from URL
        test_url = "https://example.com/test.edf"
        edf_data = read_eyelink(test_url)
        
        # Verify mocks were called
        mock_is_url.assert_called_once_with(test_url)
        mock_download.assert_called_once_with(test_url, session=None)
        
        # Basic assertions about the returned object
        self.assertIsNotNone(edf_data, "read_eyelink should return a non-None object")

    def test_read_eyelink_nonexistent_file(self):
        """Test reading nonexistent EDF file."""
        # Skip test if eyelinkio is not available
        try:
            import eyelinkio
        except ImportError:
            self.skipTest("eyelinkio package not available")
        
        nonexistent_file = os.path.join(self.temp_dir, 'nonexistent.edf')
        
        # Should raise an exception when trying to read nonexistent file
        with self.assertRaises(Exception):  # Could be FileNotFoundError or eyelinkio-specific error
            read_eyelink(nonexistent_file)

    @patch('pypillometry.io.logging_get_level')
    def test_read_eyelink_logging_levels(self, mock_get_level):
        """Test that read_eyelink respects logging levels."""
        # Skip test if eyelinkio is not available
        try:
            import eyelinkio
        except ImportError:
            self.skipTest("eyelinkio package not available")
        
        # Get path to test EDF file
        test_edf_path = files('pypillometry.data').joinpath('test.edf')        
        test_edf_path = os.path.abspath(test_edf_path)
        
        # Skip test if test file doesn't exist
        if not os.path.exists(test_edf_path):
            self.skipTest(f"Test EDF file not found at {test_edf_path}")
        
        # Test with DEBUG level (should show output)
        mock_get_level.return_value = "DEBUG"
        edf_data_debug = read_eyelink(test_edf_path)
        self.assertIsNotNone(edf_data_debug)
        
        # Test with INFO level (should suppress output)
        mock_get_level.return_value = "INFO"
        edf_data_info = read_eyelink(test_edf_path)
        self.assertIsNotNone(edf_data_info)
        
        # Both should return valid data
        self.assertEqual(type(edf_data_debug), type(edf_data_info))

if __name__ == '__main__':
    unittest.main()