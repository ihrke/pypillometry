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
        pass
        #d=pp.PupilData.from_file("data/test.pd")
        #self.assertEqual(d.__class__, pp.PupilData)
    def test_scale(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        d=d.scale("pupil")
        self.assertEqual(d.data["left","pupil"].mean(), 0)
        self.assertEqual(d.data["right","pupil"].mean(), 0)
        self.assertEqual(d.scale_params["left","pupil","mean"], 1.5)
        self.assertEqual(d.scale_params["right","pupil","mean"], 3.5)
        self.assertEqual(d.scale_params["left","pupil","sd"], 0.5)
        self.assertEqual(d.scale_params["right","pupil","sd"], 0.5)
    def test_scale_fixed(self):
        d = pp.PupilData(sampling_rate=10, left_pupil=[1,2,1,2], right_pupil=[3,4,3,4])
        scalepars = pp.Parameters({("mean","left","pupil"):1, 
                                   ("mean","right","pupil"):3, 
                                   ("sd","left","pupil"):0.5, 
                                   ("sd","right","pupil"):0.5})
        d=d.scale("pupil", mean=scalepars)
        

if __name__ == '__main__':
    unittest.main()