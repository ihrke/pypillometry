import unittest
import sys
#sys.path.insert(0,"..")
#import pypillometry as pp
from .. import *

class TestPupil(unittest.TestCase):
    def setUp(self):
        self.d=create_fake_pupildata(ntrials=100)
    def test_pupil_kernel_t(self):
        pupil_kernel_t([1,2], 10, 900)

if __name__ == '__main__':
    unittest.main()