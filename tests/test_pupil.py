import unittest
import sys
import pypillometry as pp
from pypillometry.signal.pupil import pupil_kernel_t

class TestPupil(unittest.TestCase):
    def setUp(self):
        self.d = pp.get_example_data("rlmw_002_short")
        
    def test_pupil_kernel_t(self):
        pupil_kernel_t([1,2], 10, 900)

if __name__ == '__main__':
    unittest.main()