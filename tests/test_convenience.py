import unittest
import numpy as np
from pypillometry.convenience import ByteSize, sizeof_fmt

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

if __name__ == '__main__':
    unittest.main() 