import unittest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from pypillometry.convenience import ByteSize, sizeof_fmt, is_url, normalize_unit, UNIT_ALIASES, CANONICAL_UNITS
from pypillometry.io import download

class TestConvenience(unittest.TestCase):
    def test_byte_size_basic(self):
        """Test basic ByteSize functionality."""
        # Test simple integer initialization
        size = ByteSize(1024)
        self.assertEqual(int(size), 1024)
        self.assertEqual(str(size), "1.0KiB")
        self.assertEqual(size.cached_bytes, 0)
        
        # Test dictionary initialization
        size = ByteSize({'memory': 1024, 'disk': 512})
        self.assertEqual(int(size), 1536)  # 1024 + 512
        self.assertEqual(str(size), "1.0KiB (+512.0B cached)")
        self.assertEqual(size.cached_bytes, 512)
        
    def test_byte_size_arithmetic(self):
        """Test arithmetic operations with ByteSize."""
        size1 = ByteSize(1024)
        size2 = ByteSize(2048)
        
        # Addition
        self.assertEqual(size1 + size2, 3072)
        self.assertEqual(size1 + 1024, 2048)
        self.assertEqual(1024 + size1, 2048)
        
        # Subtraction
        self.assertEqual(size2 - size1, 1024)
        self.assertEqual(size2 - 1024, 1024)
        self.assertEqual(3072 - size1, 2048)
        
        # Multiplication
        self.assertEqual(size1 * 2, 2048)
        self.assertEqual(2 * size1, 2048)
        
        # Division
        self.assertEqual(size2 / 2, 1024)
        self.assertEqual(size2 / size1, 2)
        
    def test_byte_size_comparison(self):
        """Test comparison operations with ByteSize."""
        size1 = ByteSize(1024)
        size2 = ByteSize(2048)
        
        self.assertTrue(size1 < size2)
        self.assertTrue(size1 <= size2)
        self.assertTrue(size2 > size1)
        self.assertTrue(size2 >= size1)
        self.assertTrue(size1 == 1024)
        self.assertTrue(size1 != size2)
        
    def test_byte_size_cached(self):
        """Test cached ByteSize functionality."""
        # Test with cached bytes
        size1 = ByteSize({'memory': 1024, 'disk': 256})
        self.assertEqual(size1.cached_bytes, 256)
        self.assertEqual(int(size1), 1280)  # 1024 + 256
        
        # Test arithmetic with cached sizes
        size2 = size1 + 512
        self.assertIsInstance(size2, int)  # Result should be a regular integer
        self.assertEqual(size2, 1792)  # (1024 + 256) + 512 = 1792
        
        # Test that cached bytes are preserved in the original object
        self.assertEqual(size1.cached_bytes, 256)  # Original object's cached bytes preserved
        
        # Test with another ByteSize object
        size3 = ByteSize({'memory': 2048, 'disk': 512})
        self.assertEqual(int(size3), 2560)  # 2048 + 512
        size4 = size1 + size3
        self.assertIsInstance(size4, int)  # Result should be a regular integer
        self.assertEqual(size4, 3840)  # (1024 + 256) + (2048 + 512) = 3840
        self.assertEqual(size3.cached_bytes, 512)  # Original object's cached bytes preserved
        
    def test_byte_size_edge_cases(self):
        """Test edge cases for ByteSize."""
        # Test zero size
        size = ByteSize(0)
        self.assertEqual(int(size), 0)
        self.assertEqual(str(size), "0.0B")
        
        # Test large size
        size = ByteSize(1024**4)  # 1 TiB
        self.assertEqual(str(size), "1.0TiB")
        
        # Test with empty cache
        size = ByteSize({'memory': 1024, 'disk': 0})
        self.assertEqual(str(size), "1.0KiB")
        
    def test_sizeof_fmt(self):
        """Test the sizeof_fmt function."""
        self.assertEqual(sizeof_fmt(0), "0.0B")
        self.assertEqual(sizeof_fmt(1024), "1.0KiB")
        self.assertEqual(sizeof_fmt(1024**2), "1.0MiB")
        self.assertEqual(sizeof_fmt(1024**3), "1.0GiB")
        self.assertEqual(sizeof_fmt(1024**4), "1.0TiB")
        
        # Test negative numbers
        self.assertEqual(sizeof_fmt(-1024), "-1.0KiB")
        
        # Test non-power-of-2 numbers
        self.assertEqual(sizeof_fmt(1500), "1.5KiB")
        self.assertEqual(sizeof_fmt(1536), "1.5KiB")

    @patch('tqdm.tqdm')
    def test_download_with_filename(self, mock_tqdm):
        """Test download function with specified filename."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.headers.get.return_value = '1024'  # Content-length
        mock_response.iter_content.return_value = [b'test data chunk 1', b'test data chunk 2']
        mock_response.raise_for_status = MagicMock()
        mock_response_context = MagicMock()
        mock_response_context.__enter__.return_value = mock_response
        mock_response_context.__exit__.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response_context
        
        # Mock tqdm
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress_bar
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            # Test download with specified filename
            result = download('https://example.com/test.txt', fname=tmp_filename, session=mock_session)
            
            # Verify the result
            self.assertEqual(result, tmp_filename)
            
            # Verify session.get was called correctly
            mock_session.get.assert_called_once_with('https://example.com/test.txt', stream=True)
            mock_session.close.assert_not_called()
            
            # Verify file was written
            self.assertTrue(os.path.exists(tmp_filename))
            with open(tmp_filename, 'rb') as f:
                content = f.read()
                self.assertEqual(content, b'test data chunk 1test data chunk 2')
                
        finally:
            # Clean up
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    @patch('tqdm.tqdm')
    @patch('tempfile.mkstemp')
    def test_download_without_filename(self, mock_mkstemp, mock_tqdm):
        """Test download function without filename (creates temporary file)."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.headers.get.return_value = '1024'  # Content-length
        mock_response.iter_content.return_value = [b'test data chunk 1', b'test data chunk 2']
        mock_response.raise_for_status = MagicMock()
        mock_response_context = MagicMock()
        mock_response_context.__enter__.return_value = mock_response
        mock_response_context.__exit__.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response_context
        
        # Mock tqdm
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress_bar
        
        # Create a real temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_filename = tmp_file.name
        
        # Mock mkstemp to return a valid file descriptor and filename
        import os
        tmp_fd = os.open(tmp_filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        mock_mkstemp.return_value = (tmp_fd, tmp_filename)
        
        try:
            # Test download without filename
            result = download('https://example.com/test.txt', fname=None, session=mock_session)
            
            # Verify the result
            self.assertEqual(result, tmp_filename)
            
            # Verify mkstemp was called with correct suffix
            mock_mkstemp.assert_called_once_with(suffix='.txt')
            
            # Verify session.get was called correctly
            mock_session.get.assert_called_once_with('https://example.com/test.txt', stream=True)
            
            # Verify file was written
            self.assertTrue(os.path.exists(tmp_filename))
            with open(tmp_filename, 'rb') as f:
                content = f.read()
                self.assertEqual(content, b'test data chunk 1test data chunk 2')
                
        finally:
            # Clean up
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    @patch('tqdm.tqdm')
    @patch('tempfile.mkstemp')
    def test_download_without_filename_no_extension(self, mock_mkstemp, mock_tqdm):
        """Test download function without filename and no file extension in URL."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.headers.get.return_value = '1024'  # Content-length
        mock_response.iter_content.return_value = [b'test data']
        mock_response.raise_for_status = MagicMock()
        mock_response_context = MagicMock()
        mock_response_context.__enter__.return_value = mock_response
        mock_response_context.__exit__.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response_context
        
        # Mock tqdm
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress_bar
        
        # Create a real temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        # Mock mkstemp to return a valid file descriptor and filename
        import os
        tmp_fd = os.open(tmp_filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        mock_mkstemp.return_value = (tmp_fd, tmp_filename)
        
        try:
            # Test download without filename and no extension in URL
            result = download('https://example.com/test', fname=None, session=mock_session)
            
            # Verify the result
            self.assertEqual(result, tmp_filename)
            
            # Verify mkstemp was called with empty suffix
            mock_mkstemp.assert_called_once_with(suffix='')
            
            # Verify session.get was called correctly
            mock_session.get.assert_called_once_with('https://example.com/test', stream=True)
                
        finally:
            # Clean up
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    def test_is_url(self):
        """Test the is_url function."""
        
        # Valid URLs
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "https://www.example.com",
            "http://example.com/path",
            "https://example.com/path/to/file.txt",
            "https://example.com:8080",
            "https://example.com:8080/path",
            "http://subdomain.example.com",
            "https://example.com/path?query=value",
            "https://example.com/path#fragment",
            "https://example.com/path?query=value#fragment",
            "ftp://ftp.example.com",
            "https://192.168.1.1",
            "http://localhost:3000",
            "https://osf.io/trsuq/download",  # Real example from codebase
        ]
        
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertTrue(is_url(url), f"'{url}' should be recognized as a valid URL")
        
        # Invalid URLs
        invalid_urls = [
            "not_a_url",
            "example.com",  # Missing scheme
            "http://",      # Missing netloc
            "://example.com",  # Missing scheme
            "file.txt",
            "/path/to/file",
            "C:\\path\\to\\file",
            "",
            "   ",
            # Note: "http:// example.com" is actually considered valid by urlparse
            # Note: "ht tp://example.com" has invalid scheme but urlparse is permissive
        ]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertFalse(is_url(url), f"'{url}' should NOT be recognized as a valid URL")
        
        # Edge cases
        edge_cases = [
            (None, False),  # None input
            (123, False),   # Integer input
            ([], False),    # List input
            ({}, False),    # Dict input
        ]
        
        for input_val, expected in edge_cases:
            with self.subTest(input=input_val):
                result = is_url(input_val)
                self.assertEqual(result, expected, f"is_url({input_val}) should return {expected}")

    def test_is_url_type_conversion(self):
        """Test that is_url handles type conversion correctly."""
        # The function converts non-strings to strings
        # Test that it doesn't crash on various types
        test_cases = [
            (123, False),
            (12.34, False),
            (True, False),
            (False, False),
            ([], False),
            ({}, False),
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                # Should not raise an exception
                result = is_url(input_val)
                self.assertEqual(result, expected)

    def test_is_url_urlparse_edge_cases(self):
        """Test edge cases where urlparse behavior might be surprising."""
        # These are cases where urlparse is more permissive than we might expect
        edge_cases = [
            ("http:// example.com", True),   # Space after scheme - urlparse accepts this
            ("ht tp://example.com", False),  # Space in scheme - urlparse treats as path
            ("http://", False),              # Missing netloc
            ("://example.com", False),       # Missing scheme
            ("http://example", True),        # No TLD - but still valid according to urlparse
            ("ftp://192.168.1.1", True),    # IP address
            ("custom://example.com", True),  # Custom scheme
        ]
        
        for url, expected in edge_cases:
            with self.subTest(url=url):
                result = is_url(url)
                self.assertEqual(result, expected, 
                    f"is_url('{url}') returned {result}, expected {expected}")


class TestNormalizeUnit(unittest.TestCase):
    """Test the normalize_unit function."""
    
    def test_canonical_units_passthrough(self):
        """Canonical units should be returned unchanged."""
        for unit in CANONICAL_UNITS:
            with self.subTest(unit=unit):
                self.assertEqual(normalize_unit(unit), unit)
    
    def test_alias_normalization(self):
        """All aliases should normalize to their canonical form."""
        test_cases = [
            # seconds
            ("s", "sec"),
            ("sec", "sec"),
            ("secs", "sec"),
            ("second", "sec"),
            ("seconds", "sec"),
            # milliseconds
            ("ms", "ms"),
            ("millisecond", "ms"),
            ("milliseconds", "ms"),
            # minutes
            ("m", "min"),
            ("min", "min"),
            ("mins", "min"),
            ("minute", "min"),
            ("minutes", "min"),
            # hours
            ("h", "h"),
            ("hr", "h"),
            ("hrs", "h"),
            ("hour", "h"),
            ("hours", "h"),
        ]
        
        for alias, expected in test_cases:
            with self.subTest(alias=alias):
                self.assertEqual(normalize_unit(alias), expected)
    
    def test_case_insensitive(self):
        """Unit normalization should be case-insensitive."""
        test_cases = [
            ("SEC", "sec"),
            ("Seconds", "sec"),
            ("MS", "ms"),
            ("Min", "min"),
            ("HOURS", "h"),
        ]
        
        for input_unit, expected in test_cases:
            with self.subTest(input=input_unit):
                self.assertEqual(normalize_unit(input_unit), expected)
    
    def test_whitespace_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        test_cases = [
            (" sec ", "sec"),
            ("\tms\t", "ms"),
            (" minutes ", "min"),
            ("  h  ", "h"),
        ]
        
        for input_unit, expected in test_cases:
            with self.subTest(input=input_unit):
                self.assertEqual(normalize_unit(input_unit), expected)
    
    def test_none_handling(self):
        """Test that None input returns None."""
        self.assertIsNone(normalize_unit(None))
    
    def test_invalid_unit_raises(self):
        """Invalid units should raise ValueError."""
        invalid_units = [
            "seconds123",
            "invalid",
            "xyz",
            "microseconds",
            "nanoseconds",
            "",
        ]
        
        for invalid in invalid_units:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError) as cm:
                    normalize_unit(invalid)
                self.assertIn("Unknown unit", str(cm.exception))
    
    def test_error_message_helpful(self):
        """Error messages should list valid units."""
        with self.assertRaises(ValueError) as cm:
            normalize_unit("invalid")
        
        error_msg = str(cm.exception)
        # Check that all canonical units are mentioned
        for unit in CANONICAL_UNITS:
            self.assertIn(unit, error_msg)
    
    def test_all_aliases_covered(self):
        """Ensure all defined aliases actually work."""
        for alias in UNIT_ALIASES.keys():
            with self.subTest(alias=alias):
                # Should not raise
                result = normalize_unit(alias)
                # Result should be one of the canonical units
                self.assertIn(result, CANONICAL_UNITS)


if __name__ == '__main__':
    unittest.main() 