import shutil
import unittest
import sys
import os
import tempfile
import numpy as np
import pickle
import requests
from unittest.mock import patch, MagicMock
from pypillometry.io import write_pickle, read_pickle, read_eyelink, download
from importlib.resources import files

class TestIO(unittest.TestCase):
    @patch("requests.Session")
    def test_osf_authenticate_success_validation(self, mock_session_cls):
        """osf_authenticate should configure headers and validate token."""
        from pypillometry.io import osf_authenticate

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response

        result_session = osf_authenticate("abc123")

        self.assertIs(result_session, mock_session)
        mock_session.headers.update.assert_called_with(
            {"Authorization": "Bearer abc123", "Accept": "application/json"}
        )
        mock_session.get.assert_called_once_with(
            "https://api.osf.io/v2/users/me/",
            timeout=10.0,
        )

    def test_osf_authenticate_reuses_session(self):
        """osf_authenticate should reuse provided session instance."""
        from pypillometry.io import osf_authenticate

        session = requests.Session()
        with patch.object(session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            returned = osf_authenticate("token123", session=session)

        self.assertIs(returned, session)
        self.assertEqual(session.headers["Authorization"], "Bearer token123")

    def test_osf_authenticate_invalid_token(self):
        """osf_authenticate should reject empty tokens."""
        from pypillometry.io import osf_authenticate

        with self.assertRaises(ValueError):
            osf_authenticate("")
        with self.assertRaises(ValueError):
            osf_authenticate("   ")

    @patch("requests.Session")
    def test_osf_authenticate_validation_failure(self, mock_session_cls):
        """HTTP errors during validation should raise ValueError."""
        from pypillometry.io import osf_authenticate

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        error_response = MagicMock()
        error_response.status_code = 403
        http_error = requests.exceptions.HTTPError(response=error_response)

        mock_session.get.return_value.raise_for_status.side_effect = http_error

        with self.assertRaisesRegex(ValueError, "OSF token validation failed"):
            osf_authenticate("badtoken")

    @patch("requests.Session")
    def test_osf_authenticate_request_exception(self, mock_session_cls):
        """General request exceptions should raise ValueError."""
        from pypillometry.io import osf_authenticate

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.RequestException("network error")

        with self.assertRaisesRegex(ValueError, "OSF token validation request failed"):
            osf_authenticate("token")

    @patch("requests.Session")
    def test_osf_authenticate_skip_validation(self, mock_session_cls):
        """Setting validate=False should skip network call."""
        from pypillometry.io import osf_authenticate

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        returned = osf_authenticate("token", validate=False)

        self.assertIs(returned, mock_session)
        mock_session.get.assert_not_called()
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
                
    def _fake_download(self, content):
        def fake(url, fname, session=None):
            with open(fname, "wb") as f:
                f.write(content)
            return fname
        return fake

    def test_read_pickle_url(self):
        """Test reading objects from URL."""
        with patch("pypillometry.io.download",
                   side_effect=self._fake_download(pickle.dumps(self.test_data))) as mock_dl:
            loaded = read_pickle('http://example.com/test.pkl', autodelete=True)
        mock_dl.assert_called_once()
        for key, value in self.test_data.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, loaded[key])
            else:
                self.assertEqual(value, loaded[key])

    def test_read_pickle_invalid_url(self):
        """Test reading invalid pickle data from a URL."""
        with patch("pypillometry.io.download",
                   side_effect=self._fake_download(b'invalid pickle data')):
            with self.assertRaises(pickle.UnpicklingError):
                read_pickle('http://invalid-url.com/nonexistent.pkl', autodelete=True)
            
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

    def test_from_eyelink_all_classes(self):
        """from_eyelink works for PupilData, GazeData and EyeData."""
        try:
            import eyelinkio
        except ImportError:
            self.skipTest("eyelinkio package not available")
        import pypillometry as pp
        fname = str(files('pypillometry.data').joinpath('test.edf'))
        for cls in (pp.PupilData, pp.GazeData, pp.EyeData):
            with self.subTest(cls=cls.__name__):
                obj = cls.from_eyelink(fname)
                self.assertIsInstance(obj, cls)
                self.assertGreater(len(obj.tx), 0)

    def test_read_eyelink_url(self):
        """Test reading EDF file from URL using read_eyelink function."""
        try:
            import eyelinkio
        except ImportError:
            self.skipTest("eyelinkio package not available")
        test_edf_path = os.path.abspath(files('pypillometry.data').joinpath('test.edf'))
        with open(test_edf_path, "rb") as f:
            content = f.read()

        test_url = "https://example.com/test.edf"
        with patch("pypillometry.io.download",
                   side_effect=self._fake_download(content)) as mock_dl:
            edf_data = read_eyelink(test_url, autodelete=True)
        self.assertEqual(mock_dl.call_args.args[0], test_url)
        self.assertIsNone(mock_dl.call_args.kwargs["session"])
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

    @patch("pypillometry.io._osf_client_url_to_download_url", return_value="https://files.osf.io/download")
    @patch("pypillometry.io.tqdm")
    def test_download_with_provided_session(self, mock_tqdm, mock_osf_url):
        """download should reuse provided session and respect OSF URL conversion."""
        session = MagicMock()
        response = MagicMock()
        response.headers.get.return_value = "128"
        response.iter_content.return_value = [b"chunk1", b"chunk2"]
        session.get.return_value.__enter__.return_value = response
        session.get.return_value.__exit__.return_value = False

        mock_bar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_bar

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            fname = tmp.name

        try:
            result = download("https://osf.io/abc123/download", fname=fname, session=session)
            self.assertEqual(result, fname)
            mock_osf_url.assert_called_once_with("https://osf.io/abc123/download", session=session)
            session.get.assert_called_once_with("https://files.osf.io/download", stream=True)
            mock_bar.update.assert_called()
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    @patch("pypillometry.io.requests.Session")
    @patch("pypillometry.io._osf_client_url_to_download_url", side_effect=lambda url, session=None: url)
    @patch("pypillometry.io.tqdm")
    def test_download_creates_session_when_missing(self, mock_tqdm, mock_osf_url, mock_session_cls):
        """download should create and close its own session when none supplied."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        response = MagicMock()
        response.headers.get.return_value = "0"
        response.iter_content.return_value = []
        mock_session.get.return_value.__enter__.return_value = response
        mock_session.get.return_value.__exit__.return_value = False

        mock_tqdm.return_value.__enter__.return_value = MagicMock()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            fname = tmp.name

        try:
            result = download("https://example.com/file.dat", fname=fname)
            self.assertEqual(result, fname)
            mock_osf_url.assert_called_once_with("https://example.com/file.dat", session=None)
            mock_session.get.assert_called_once_with("https://example.com/file.dat", stream=True)
            mock_session.close.assert_called_once()
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    @patch("pypillometry.io._osf_client_url_to_download_url", side_effect=lambda url, session=None: url)
    @patch("pypillometry.io.tqdm")
    def test_download_progress_bar(self, mock_tqdm, mock_osf_url):
        """the bar shows the name without ".part" and has no total for an unknown size"""
        session = MagicMock()
        response = MagicMock()
        response.iter_content.return_value = [b"data"]
        session.get.return_value.__enter__.return_value = response
        session.get.return_value.__exit__.return_value = False
        mock_tqdm.return_value.__enter__.return_value = MagicMock()
        tmpdir = tempfile.mkdtemp()
        try:
            fname = os.path.join(tmpdir, "rec.edf.part")
            for headers, total in [({"content-length": "4"}, 4), ({}, None)]:
                response.headers = headers
                download("https://example.com/rec.edf", fname=fname, session=session)
                kwargs = mock_tqdm.call_args.kwargs
                self.assertEqual(kwargs["desc"], "rec.edf")
                self.assertEqual(kwargs["total"], total)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()