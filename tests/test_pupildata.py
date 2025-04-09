import unittest
import sys
import numpy as np
sys.path.insert(0,"..")
import pypillometry as pp
import pytest

class TestPupilData(unittest.TestCase):
    def setUp(self):
        #self.d=pp.PupilData.from_file("data/test.pd")
        pass
    def test_from_file(self):
        d = pp.get_example_data("rlmw_002_short")
        print(d)
    def test_scale(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        d=d.scale("pupil")
        self.assertEqual(d.data["left","pupil"].mean(), 0)
        self.assertEqual(d.data["right","pupil"].mean(), 0)
        self.assertEqual(d.params["scale"]["mean"]["left"]["pupil"], 1.5)
        self.assertEqual(d.params["scale"]["mean"]["right"]["pupil"], 3.5)
    def test_scale_fixed(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        scalepars = {
            "left": {"pupil": 1},
            "right": {"pupil": 3}
        }
        sdpars = {
            "left": {"pupil": 0.5},
            "right": {"pupil": 0.5}
        }
        d=d.scale("pupil", mean=scalepars, sd=sdpars)
        self.assertEqual(d.data["left","pupil"].mean(), 1)
        self.assertEqual(d.data["right","pupil"].mean(), 1)
    def test_unscale(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        d=d.scale("pupil")
        d=d.unscale("pupil")
        self.assertEqual(d.data["left","pupil"].mean(), 1.5)
        self.assertEqual(d.data["right","pupil"].mean(), 3.5)
        

if __name__ == '__main__':
    unittest.main()