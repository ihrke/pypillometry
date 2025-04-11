import unittest
import sys
import os
import tempfile
import numpy as np
import pickle
import requests
from unittest.mock import patch, MagicMock
from pypillometry.io import write_pickle, read_pickle

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

if __name__ == '__main__':
    unittest.main()